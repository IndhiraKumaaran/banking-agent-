"""One guarded LangChain agent for the mock banking support workflows."""

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool, tool
from langchain_google_genai import ChatGoogleGenerativeAI

from app import tools as banking_tools

PROJECT_ROOT = Path(__file__).resolve().parents[1]


SYSTEM_PROMPT = """You are the single assistant for a fictional banking support demo.
Use the available tools for calculations, eligibility, mock account retrieval,
and application preparation. Never invent tool results or financial values.

For a loan request, call calculate_emi first, then check_loan_eligibility with
that exact EMI, then prepare_loan_application. The eligibility tool refuses to
run until the matching EMI tool has run. The preparation tool repeats the
calculation and eligibility checks in deterministic Python.

If the customer is ineligible, explain the tool result and do not ask for
confirmation. If eligible, display every value in the preparation summary and
ask the customer to reply exactly: "I confirm the loan application details
exactly as shown." Never treat the initial request, "yes", or a vague reply as
confirmation. Do not claim an application was submitted unless a tool result
says it was. Account retrieval is limited to the identity supplied by the API;
never ask for or expose passwords, PINs, OTPs, CVVs, or full account numbers.
All records are fictional and this is not production banking software."""

EXPLICIT_CONFIRMATION_PHRASES = {
	"i confirm the loan application details exactly as shown",
	"i explicitly confirm the loan application details exactly as shown",
}


class AgentConfigurationError(RuntimeError):
	"""Raised when the chat model cannot be configured safely."""


class AgentProviderError(RuntimeError):
	"""Raised when a chat model invocation fails."""


class AgentAuthorizationError(PermissionError):
	"""Raised when a conversation is used by a different demo identity."""


@dataclass
class AgentConversation:
	"""Per-process conversation state bound to one simulated customer identity."""

	conversation_id: str
	authenticated_customer_id: str
	messages: list[BaseMessage] = field(default_factory=list)
	pending_application: dict[str, Any] | None = None
	pending_confirmation_token: str | None = None
	last_application_result: dict[str, Any] | None = None
	last_eligibility_result: dict[str, Any] | None = None


AGENT_CONVERSATIONS: dict[str, AgentConversation] = {}


def _json_result(value: Any) -> str:
	return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _format_money(value: float) -> str:
	return f"{value:,.2f}"


def create_banking_tools(
	authenticated_customer_id: str | None,
	conversation: AgentConversation | None = None,
	*,
	explicit_confirmation: bool = False,
) -> list[BaseTool]:
	"""Create the five banking tools, binding account access to the API identity."""
	calculation_state: dict[str, float | bool | None] = {
		"emi_ready": False,
		"monthly_emi": None,
	}

	@tool("calculate_emi")
	def calculate_emi_tool(
		loan_amount: float,
		annual_interest_rate: float,
		tenure_months: int,
	) -> str:
		"""Calculate the monthly EMI and repayment totals for a loan request."""
		try:
			result = banking_tools.calculate_emi(
				loan_amount,
				annual_interest_rate,
				tenure_months,
			)
		except ValueError as error:
			calculation_state["emi_ready"] = False
			return _json_result({"error": str(error)})
		calculation_state["emi_ready"] = True
		calculation_state["monthly_emi"] = result["monthly_emi"]
		return _json_result(result)

	@tool("check_loan_eligibility")
	def check_loan_eligibility_tool(
		monthly_salary: float,
		existing_monthly_emi: float,
		proposed_monthly_emi: float,
	) -> str:
		"""Check affordability after calculating the matching proposed EMI."""
		if not calculation_state["emi_ready"]:
			return _json_result({"error": "Call calculate_emi first for this loan request."})
		if round(proposed_monthly_emi, 2) != calculation_state["monthly_emi"]:
			calculation_state["emi_ready"] = False
			return _json_result({
				"error": "The proposed EMI must match the most recently calculated EMI."
			})
		calculation_state["emi_ready"] = False
		try:
			result = banking_tools.check_loan_eligibility(
				monthly_salary,
				existing_monthly_emi,
				float(calculation_state["monthly_emi"]),
			)
		except ValueError as error:
			return _json_result({"error": str(error)})
		if conversation is not None:
			conversation.last_eligibility_result = result
		return _json_result(result)

	@tool("get_account_details")
	def get_account_details_tool(customer_id: str) -> str:
		"""Retrieve masked mock account data for the authenticated customer only."""
		try:
			result = banking_tools.get_account_details(
				customer_id,
				authenticated_customer_id,
			)
		except (PermissionError, LookupError) as error:
			return _json_result({"error": str(error)})
		return _json_result(result)

	@tool("prepare_loan_application")
	def prepare_loan_application_tool(
		loan_amount: float,
		tenure_months: int,
		annual_interest_rate: float,
		monthly_salary: float,
		existing_monthly_emi: float,
	) -> str:
		"""Calculate, check eligibility, and prepare exact details for confirmation."""
		if not authenticated_customer_id:
			return _json_result({"error": "Authenticated customer identity is required."})
		try:
			emi_result = banking_tools.calculate_emi(
				loan_amount,
				annual_interest_rate,
				tenure_months,
			)
			application_details = {
				"customer_id": authenticated_customer_id,
				"loan_amount": loan_amount,
				"tenure_months": tenure_months,
				"annual_interest_rate": annual_interest_rate,
				"monthly_salary": monthly_salary,
				"existing_monthly_emi": existing_monthly_emi,
				"proposed_monthly_emi": emi_result["monthly_emi"],
			}
			result = banking_tools.submit_loan_application(
				**application_details,
				confirmed=False,
			)
		except ValueError as error:
			return _json_result({"error": str(error)})

		if conversation is not None:
			conversation.last_application_result = result
			if result["status"] == "confirmation_required":
				conversation.pending_application = application_details
				conversation.pending_confirmation_token = result["confirmation_token"]
			else:
				conversation.pending_application = None
				conversation.pending_confirmation_token = None

		# The API holds the token; do not expose it to the model's context.
		model_result = {key: value for key, value in result.items() if key != "confirmation_token"}
		return _json_result(model_result)

	@tool("submit_confirmed_application")
	def submit_confirmed_application_tool() -> str:
		"""Submit only after the API verifies an exact customer confirmation phrase."""
		if not explicit_confirmation:
			return _json_result({"error": "Explicit customer confirmation is required."})
		if (
			conversation is None
			or conversation.pending_application is None
			or conversation.pending_confirmation_token is None
		):
			return _json_result({"error": "There is no pending application to confirm."})
		if conversation.pending_application["customer_id"] != authenticated_customer_id:
			return _json_result({"error": "The pending application belongs to another identity."})

		try:
			result = banking_tools.submit_loan_application(
				**conversation.pending_application,
				confirmed=True,
				confirmation_token=conversation.pending_confirmation_token,
			)
		except ValueError as error:
			return _json_result({"error": str(error)})
		if result["status"] in {"submitted", "already_submitted"}:
			conversation.pending_application = None
			conversation.pending_confirmation_token = None
		return _json_result(result)

	return [
		calculate_emi_tool,
		check_loan_eligibility_tool,
		get_account_details_tool,
		prepare_loan_application_tool,
		submit_confirmed_application_tool,
	]


def create_banking_agent(
	authenticated_customer_id: str,
	conversation: AgentConversation,
	*,
	model: Any | None = None,
	explicit_confirmation: bool = False,
) -> tuple[Any, list[BaseTool]]:
	"""Build one LangChain agent; accept an injected chat model for offline tests."""
	if model is None:
		load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
		api_key = os.getenv("GOOGLE_API_KEY", "").strip()
		if not api_key or api_key == "your-google-api-key-here":
			raise AgentConfigurationError(
				"The agent is not configured. Set GOOGLE_API_KEY to enable chat."
			)
		model_name = os.getenv("GEMINI_MODEL", "").strip() or "gemini-2.5-flash"
		try:
			model = ChatGoogleGenerativeAI(
				model=model_name,
				google_api_key=api_key,
				temperature=0,
			)
		except Exception:
			raise AgentConfigurationError(
				"The configured chat model could not be initialized."
			) from None

	tools = create_banking_tools(
		authenticated_customer_id,
		conversation,
		explicit_confirmation=explicit_confirmation,
	)
	try:
		agent = create_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT)
	except Exception:
		raise AgentProviderError("The chat model could not be initialized.") from None
	return agent, tools


def _is_explicit_confirmation(message: str) -> bool:
	normalized = " ".join(message.casefold().strip().split()).rstrip(".!?")
	return normalized in EXPLICIT_CONFIRMATION_PHRASES


def _format_money(value: float) -> str:
	return f"{value:,.2f}"


def _application_summary_message(result: dict[str, Any]) -> str:
	summary = result["application_summary"]
	return (
		"Eligibility check passed under the demo's illustrative rules. Review these "
		"exact details: "
		f"loan amount Rs. {_format_money(summary['loan_amount'])}; "
		f"tenure {summary['tenure_months']} months; "
		f"annual interest rate {summary['annual_interest_rate']}%; "
		f"proposed monthly EMI Rs. {_format_money(summary['proposed_monthly_emi'])}; "
		f"monthly salary Rs. {_format_money(summary['monthly_salary'])}; "
		f"existing monthly EMI Rs. {_format_money(summary['existing_monthly_emi'])}; "
		f"total monthly EMI Rs. {_format_money(summary['total_monthly_emi'])}. "
		'To submit, reply exactly: "I confirm the loan application details exactly as shown."'
	)


def run_agent_turn(
	message: str,
	authenticated_customer_id: str,
	conversation_id: str | None = None,
	*,
	model: Any | None = None,
) -> dict[str, Any]:
	"""Run one chat turn, enforcing identity and exact confirmation in Python."""
	if not authenticated_customer_id or not authenticated_customer_id.strip():
		raise AgentAuthorizationError("Authenticated customer identity is required.")
	try:
		banking_tools.get_account_details(
			authenticated_customer_id,
			authenticated_customer_id,
		)
	except PermissionError as error:
		raise AgentAuthorizationError(str(error)) from error

	if conversation_id is None:
		conversation_id = uuid.uuid4().hex
	conversation = AGENT_CONVERSATIONS.get(conversation_id)
	if conversation is None:
		conversation = AgentConversation(
			conversation_id=conversation_id,
			authenticated_customer_id=authenticated_customer_id,
		)
		AGENT_CONVERSATIONS[conversation_id] = conversation
	elif conversation.authenticated_customer_id != authenticated_customer_id:
		raise AgentAuthorizationError(
			"This conversation is bound to a different authenticated customer."
		)

	if _is_explicit_confirmation(message):
		if conversation.pending_application is None:
			return {
				"conversation_id": conversation_id,
				"status": "no_pending_application",
				"response": "There is no pending eligible application to confirm.",
			}
		tools = create_banking_tools(
			authenticated_customer_id,
			conversation,
			explicit_confirmation=True,
		)
		tool_result = tools[-1].invoke({})
		result = json.loads(tool_result)
		if result.get("status") in {"submitted", "already_submitted"}:
			response = (
				"The mock application was submitted. Application ID: "
				f"{result['application']['application_id']}."
			)
		else:
			response = result.get("error", "The application could not be submitted.")
		conversation.messages.extend(
			[HumanMessage(content=message), AIMessage(content=response)]
		)
		return {
			"conversation_id": conversation_id,
			"status": result.get("status", "error"),
			"response": response,
			"application": result.get("application"),
		}

	previous_pending = (
		conversation.pending_application,
		conversation.pending_confirmation_token,
	)
	conversation.last_application_result = None
	conversation.last_eligibility_result = None
	try:
		agent, _ = create_banking_agent(
			authenticated_customer_id,
			conversation,
			model=model,
		)
		result = agent.invoke({"messages": [*conversation.messages, HumanMessage(content=message)]})
	except AgentConfigurationError:
		conversation.pending_application, conversation.pending_confirmation_token = previous_pending
		raise
	except Exception:
		conversation.pending_application, conversation.pending_confirmation_token = previous_pending
		raise AgentProviderError(
			"The chat model provider failed. Please try again."
		) from None

	conversation.messages = result["messages"]
	application_result = conversation.last_application_result
	if application_result is not None and application_result["status"] == "ineligible":
		return {
			"conversation_id": conversation_id,
			"status": "ineligible",
			"response": (
				f"The application is ineligible: {application_result['message']} "
				"No application was submitted."
			),
			"application_summary": application_result["application_summary"],
		}
	if application_result is not None and application_result["status"] == "confirmation_required":
		return {
			"conversation_id": conversation_id,
			"status": "confirmation_required",
			"response": _application_summary_message(application_result),
			"application_summary": application_result["application_summary"],
			"confirmation_token": application_result["confirmation_token"],
		}

	eligibility_result = conversation.last_eligibility_result
	if eligibility_result is not None and not eligibility_result["eligible"]:
		return {
			"conversation_id": conversation_id,
			"status": "ineligible",
			"response": f"The customer is ineligible: {eligibility_result['reason']}",
			"eligibility": eligibility_result,
		}

	last_message = result["messages"][-1]
	content = last_message.content
	if isinstance(content, list):
		content = "".join(
			part.get("text", "") for part in content if isinstance(part, dict)
		)
	return {
		"conversation_id": conversation_id,
		"status": "response",
		"response": str(content),
	}