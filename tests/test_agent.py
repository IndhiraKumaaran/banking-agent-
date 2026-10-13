"""Offline tests for the single LangChain banking agent."""

import json

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app import agent as agent_module
from app import tools as banking_tools
from app.agent import (
	AGENT_CONVERSATIONS,
	AgentAuthorizationError,
	AgentConfigurationError,
	AgentProviderError,
	AgentConversation,
	create_banking_agent,
	create_banking_tools,
	run_agent_turn,
)
from app.mock_data import (
	MOCK_LOAN_APPLICATIONS,
	MOCK_LOAN_APPLICATIONS_BY_FINGERPRINT,
	PENDING_LOAN_CONFIRMATIONS,
)


class ScriptedToolChatModel(BaseChatModel):
	"""A no-network chat model that returns pre-scripted tool calls or text."""

	responses: list[BaseMessage]
	response_index: int = 0

	@property
	def _llm_type(self) -> str:
		return "scripted-tool-model"

	def bind_tools(self, tools, *, tool_choice=None, **kwargs):
		return self

	def _generate(
		self,
		messages: list[BaseMessage],
		stop: list[str] | None = None,
		run_manager: CallbackManagerForLLMRun | None = None,
		**kwargs,
	) -> ChatResult:
		if self.response_index >= len(self.responses):
			raise RuntimeError("The scripted model has no more responses.")
		response = self.responses[self.response_index]
		self.response_index += 1
		return ChatResult(generations=[ChatGeneration(message=response)])


@pytest.fixture(autouse=True)
def clear_agent_state():
	AGENT_CONVERSATIONS.clear()
	MOCK_LOAN_APPLICATIONS.clear()
	MOCK_LOAN_APPLICATIONS_BY_FINGERPRINT.clear()
	PENDING_LOAN_CONFIRMATIONS.clear()


def fake_tool_call(name, arguments, call_id):
	return AIMessage(
		content="",
		tool_calls=[{"name": name, "args": arguments, "id": call_id}],
	)


def loan_request_tool_calls(
	*,
	loan_amount=500_000,
	tenure_months=36,
	annual_interest_rate=12,
	monthly_salary=60_000,
	existing_monthly_emi=45_000,
):
	proposed_emi = banking_tools.calculate_emi(
		loan_amount,
		annual_interest_rate,
		tenure_months,
	)["monthly_emi"]
	return [
		fake_tool_call(
			"calculate_emi",
			{
				"loan_amount": loan_amount,
				"annual_interest_rate": annual_interest_rate,
				"tenure_months": tenure_months,
			},
			"call_emi",
		),
		fake_tool_call(
			"check_loan_eligibility",
			{
				"monthly_salary": monthly_salary,
				"existing_monthly_emi": existing_monthly_emi,
				"proposed_monthly_emi": proposed_emi,
			},
			"call_eligibility",
		),
		fake_tool_call(
			"prepare_loan_application",
			{
				"loan_amount": loan_amount,
				"tenure_months": tenure_months,
				"annual_interest_rate": annual_interest_rate,
				"monthly_salary": monthly_salary,
				"existing_monthly_emi": existing_monthly_emi,
			},
			"call_prepare",
		),
		AIMessage(content="I have checked the request."),
	]


def test_agent_registers_only_the_expected_banking_tools():
	registered_tools = create_banking_tools("CUST1001")

	assert [registered_tool.name for registered_tool in registered_tools] == [
		"calculate_emi",
		"check_loan_eligibility",
		"get_account_details",
		"prepare_loan_application",
		"submit_confirmed_application",
	]


def test_eligibility_tool_requires_a_calculated_emi_first():
	registered_tools = {
		registered_tool.name: registered_tool
		for registered_tool in create_banking_tools("CUST1001")
	}

	result = json.loads(
		registered_tools["check_loan_eligibility"].invoke(
			{
				"monthly_salary": 60_000,
				"existing_monthly_emi": 0,
				"proposed_monthly_emi": 8_333.33,
			}
		)
	)

	assert "Call calculate_emi first" in result["error"]


def test_agent_calculates_before_eligibility_and_blocks_ineligible_confirmation(monkeypatch):
	events = []
	original_calculate = banking_tools.calculate_emi
	original_eligibility = banking_tools.check_loan_eligibility

	def track_calculation(*args, **kwargs):
		events.append("emi")
		return original_calculate(*args, **kwargs)

	def track_eligibility(*args, **kwargs):
		events.append("eligibility")
		return original_eligibility(*args, **kwargs)

	monkeypatch.setattr(banking_tools, "calculate_emi", track_calculation)
	monkeypatch.setattr(banking_tools, "check_loan_eligibility", track_eligibility)
	model = ScriptedToolChatModel(
		responses=loan_request_tool_calls(existing_monthly_emi=45_000)
	)

	result = run_agent_turn(
		"I want a personal loan of Rs. 500000 for 36 months.",
		"CUST1001",
		model=model,
	)

	assert result["status"] == "ineligible"
	assert "No application was submitted" in result["response"]
	assert "confirm" not in result["response"].lower()
	assert events.index("emi") < events.index("eligibility")
	assert all(events[index - 1] == "emi" for index, event in enumerate(events) if event == "eligibility")
	assert MOCK_LOAN_APPLICATIONS == []


def test_eligible_application_waits_for_explicit_confirmation():
	model = ScriptedToolChatModel(
		responses=loan_request_tool_calls(
			loan_amount=100_000,
			tenure_months=12,
			annual_interest_rate=0,
			existing_monthly_emi=0,
		)
	)
	result = run_agent_turn("I would like a loan.", "CUST1001", model=model)

	assert result["status"] == "confirmation_required"
	assert "loan amount Rs. 100,000.00" in result["response"]
	assert result["application_summary"]["proposed_monthly_emi"] == 8_333.33
	assert result["confirmation_token"]
	assert MOCK_LOAN_APPLICATIONS == []


def test_vague_yes_does_not_submit_application():
	initial = run_agent_turn(
		"Please prepare a loan.",
		"CUST1001",
		model=ScriptedToolChatModel(
			responses=loan_request_tool_calls(
				loan_amount=100_000,
				tenure_months=12,
				annual_interest_rate=0,
				existing_monthly_emi=0,
			)
		),
	)
	response = run_agent_turn(
		"yes",
		"CUST1001",
		initial["conversation_id"],
		model=ScriptedToolChatModel(responses=[AIMessage(content="Please review the summary and confirm explicitly.")]),
	)

	assert response["status"] == "response"
	assert MOCK_LOAN_APPLICATIONS == []


def test_exact_confirmation_submits_the_bound_application_without_an_llm():
	initial = run_agent_turn(
		"Prepare an eligible loan.",
		"CUST1001",
		model=ScriptedToolChatModel(
			responses=loan_request_tool_calls(
				loan_amount=100_000,
				tenure_months=12,
				annual_interest_rate=0,
				existing_monthly_emi=0,
			)
		),
	)
	result = run_agent_turn(
		"I confirm the loan application details exactly as shown.",
		"CUST1001",
		initial["conversation_id"],
	)

	assert result["status"] == "submitted"
	assert len(MOCK_LOAN_APPLICATIONS) == 1


def test_changed_details_require_a_new_confirmation():
	initial = run_agent_turn(
		"Prepare this loan.",
		"CUST1001",
		model=ScriptedToolChatModel(
			responses=loan_request_tool_calls(
				loan_amount=100_000,
				tenure_months=12,
				annual_interest_rate=0,
				existing_monthly_emi=0,
			)
		),
	)
	changed = run_agent_turn(
		"Change the loan to 120000.",
		"CUST1001",
		initial["conversation_id"],
		model=ScriptedToolChatModel(
			responses=loan_request_tool_calls(
				loan_amount=120_000,
				tenure_months=12,
				annual_interest_rate=0,
				existing_monthly_emi=0,
			)
		),
	)

	assert changed["status"] == "confirmation_required"
	assert changed["confirmation_token"] != initial["confirmation_token"]
	assert changed["application_summary"]["loan_amount"] == 120_000.0
	assert MOCK_LOAN_APPLICATIONS == []


def test_account_tool_uses_only_the_bound_authenticated_identity():
	account_tool = next(
		registered_tool
		for registered_tool in create_banking_tools("CUST1001")
		if registered_tool.name == "get_account_details"
	)

	permitted = json.loads(account_tool.invoke({"customer_id": "CUST1001"}))
	denied = json.loads(account_tool.invoke({"customer_id": "CUST1002"}))

	assert permitted["account_number"] == "XXXX1234"
	assert "error" in denied
	assert "balance" not in denied


def test_conversation_is_bound_to_authenticated_customer():
	initial = run_agent_turn(
		"Hello",
		"CUST1001",
			model=ScriptedToolChatModel(responses=[AIMessage(content="Hello.")]),
	)

	with pytest.raises(AgentAuthorizationError, match="bound to a different"):
		run_agent_turn(
			"Hello again",
			"CUST1002",
			initial["conversation_id"],
			model=ScriptedToolChatModel(responses=[AIMessage(content="Hello.")]),
		)


def test_missing_google_key_has_a_clear_configuration_error(monkeypatch, tmp_path):
	project_root = tmp_path / "project"
	project_root.mkdir()
	monkeypatch.setattr(agent_module, "PROJECT_ROOT", project_root)
	monkeypatch.chdir(tmp_path)
	monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
	monkeypatch.delenv("GEMINI_MODEL", raising=False)
	conversation = AgentConversation("test-chat", "CUST1001")

	with pytest.raises(AgentConfigurationError, match="Set GOOGLE_API_KEY"):
		create_banking_agent("CUST1001", conversation)


def test_example_google_key_placeholder_is_not_accepted(monkeypatch, tmp_path):
	project_root = tmp_path / "project"
	project_root.mkdir()
	monkeypatch.setattr(agent_module, "PROJECT_ROOT", project_root)
	monkeypatch.setenv("GOOGLE_API_KEY", "your-google-api-key-here")

	with pytest.raises(AgentConfigurationError, match="Set GOOGLE_API_KEY"):
		create_banking_agent("CUST1001", AgentConversation("test-chat", "CUST1001"))


def test_gemini_model_uses_configured_key_and_model_without_logging_key(monkeypatch):
	configured = {}
	secret_key = "test-gemini-key-that-must-not-be-logged"

	class FakeGeminiModel:
		def __init__(self, **kwargs):
			configured.update(kwargs)

	monkeypatch.setenv("GOOGLE_API_KEY", secret_key)
	monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
	monkeypatch.setattr(agent_module, "ChatGoogleGenerativeAI", FakeGeminiModel)
	conversation = AgentConversation("test-chat", "CUST1001")

	_, tools = create_banking_agent("CUST1001", conversation)

	assert configured == {
		"model": "gemini-2.5-flash",
		"google_api_key": secret_key,
		"temperature": 0,
	}
	assert len(tools) == 5


def test_project_root_dotenv_loads_before_model_configuration(monkeypatch, tmp_path):
	project_root = tmp_path / "project"
	working_directory = tmp_path / "working"
	project_root.mkdir()
	working_directory.mkdir()
	(project_root / ".env").write_text("GOOGLE_API_KEY=project-test-key\n")
	(working_directory / ".env").write_text("GOOGLE_API_KEY=working-test-key\n")
	monkeypatch.setattr(agent_module, "PROJECT_ROOT", project_root)
	monkeypatch.chdir(working_directory)
	monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
	monkeypatch.delenv("GEMINI_MODEL", raising=False)

	configured = {}

	class FakeGeminiModel:
		def __init__(self, **kwargs):
			configured.update(kwargs)

	monkeypatch.setattr(agent_module, "ChatGoogleGenerativeAI", FakeGeminiModel)
	monkeypatch.setattr(agent_module, "create_agent", lambda **kwargs: kwargs)

	create_banking_agent("CUST1001", AgentConversation("test-chat", "CUST1001"))

	assert configured == {
		"model": "gemini-2.5-flash",
		"google_api_key": "project-test-key",
		"temperature": 0,
	}


def test_project_root_dotenv_configures_gemini_model(monkeypatch, tmp_path):
	project_root = tmp_path / "project"
	working_directory = tmp_path / "working"
	project_root.mkdir()
	working_directory.mkdir()
	(project_root / ".env").write_text(
		"GOOGLE_API_KEY=project-test-key\nGEMINI_MODEL=gemini-test-model\n"
	)
	monkeypatch.setattr(agent_module, "PROJECT_ROOT", project_root)
	monkeypatch.chdir(working_directory)
	monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
	monkeypatch.delenv("GEMINI_MODEL", raising=False)

	configured = {}

	class FakeGeminiModel:
		def __init__(self, **kwargs):
			configured.update(kwargs)

	monkeypatch.setattr(agent_module, "ChatGoogleGenerativeAI", FakeGeminiModel)
	monkeypatch.setattr(agent_module, "create_agent", lambda **kwargs: kwargs)

	create_banking_agent("CUST1001", AgentConversation("test-chat", "CUST1001"))

	assert configured == {
		"model": "gemini-test-model",
		"google_api_key": "project-test-key",
		"temperature": 0,
	}


def test_model_failure_is_sanitized_and_does_not_expose_provider_error(monkeypatch):
	class FailingAgent:
		def invoke(self, inputs):
			raise RuntimeError("provider error contains api-key=private")

	monkeypatch.setattr(agent_module, "create_agent", lambda **kwargs: FailingAgent())

	with pytest.raises(AgentProviderError, match="provider failed") as error:
		run_agent_turn(
			"Hello",
			"CUST1001",
			model=ScriptedToolChatModel(responses=[AIMessage(content="unused")]),
		)

	assert "private" not in str(error.value)