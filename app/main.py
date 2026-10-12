"""FastAPI endpoints for the mock banking support tools."""

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.agent import (
	AgentAuthorizationError,
	AgentConfigurationError,
	AgentProviderError,
	run_agent_turn,
)
from app.tools import (
	check_loan_eligibility,
	get_account_details,
	submit_loan_application,
)

app = FastAPI(title="Smart Banking Support Agent API")


class LoanEligibilityRequest(BaseModel):
	"""Validated input for the illustrative loan eligibility check."""

	monthly_salary: float = Field(gt=0, allow_inf_nan=False)
	existing_monthly_emi: float = Field(ge=0, allow_inf_nan=False)
	proposed_monthly_emi: float = Field(ge=0, allow_inf_nan=False)
	max_emi_ratio: float = Field(default=0.50, gt=0, le=1, allow_inf_nan=False)


class LoanApplicationRequest(BaseModel):
	"""Validated input for a mock loan application submission."""

	customer_id: str = Field(min_length=1)
	loan_amount: float = Field(gt=0, allow_inf_nan=False)
	tenure_months: int = Field(gt=0)
	annual_interest_rate: float = Field(ge=0, allow_inf_nan=False)
	monthly_salary: float = Field(gt=0, allow_inf_nan=False)
	existing_monthly_emi: float = Field(ge=0, allow_inf_nan=False)
	proposed_monthly_emi: float = Field(ge=0, allow_inf_nan=False)
	confirmed: bool = False
	confirmation_token: str | None = None


class AgentChatRequest(BaseModel):
	"""A user message and optional ID for an in-memory chat conversation."""

	message: str = Field(min_length=1)
	conversation_id: str | None = None


@app.exception_handler(RequestValidationError)
async def validation_error_response(
	request: Request,
	error: RequestValidationError,
) -> JSONResponse:
	"""Return validation details without echoing submitted values."""
	return JSONResponse(
		status_code=422,
		content={
			"detail": [
				{
					"field": ".".join(str(part) for part in error["loc"]),
					"message": error["msg"],
				}
				for error in error.errors()
			],
		},
	)


@app.get("/health")
def health_check() -> dict[str, str]:
	"""Return a simple status response to show the API is running."""
	return {"status": "ok"}


@app.get("/accounts/{customer_id}")
def account_details(
	customer_id: str,
	authenticated_customer_id: str | None = Header(
		default=None,
		alias="X-Authenticated-Customer-ID",
	),
) -> dict[str, str | float]:
	"""Return a customer's mock account details after the tool's identity check."""
	try:
		return get_account_details(customer_id, authenticated_customer_id)
	except PermissionError as error:
		status_code = 401 if authenticated_customer_id is None else 403
		raise HTTPException(status_code=status_code, detail=str(error)) from error
	except LookupError as error:
		raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/loan/eligibility")
def loan_eligibility(
	request: LoanEligibilityRequest,
) -> dict[str, bool | float | str]:
	"""Return the existing tool's eligibility result for validated input."""
	return check_loan_eligibility(
		monthly_salary=request.monthly_salary,
		existing_monthly_emi=request.existing_monthly_emi,
		proposed_monthly_emi=request.proposed_monthly_emi,
		max_emi_ratio=request.max_emi_ratio,
	)


@app.post(
	"/loan/applications",
	responses={201: {"description": "Application submitted"}, 202: {"description": "Customer confirmation required"}},
)
def loan_application(
	request: LoanApplicationRequest,
) -> JSONResponse:
	"""Prepare an application or submit it after matching confirmation."""
	try:
		result = submit_loan_application(
			customer_id=request.customer_id,
			loan_amount=request.loan_amount,
			tenure_months=request.tenure_months,
			annual_interest_rate=request.annual_interest_rate,
			monthly_salary=request.monthly_salary,
			existing_monthly_emi=request.existing_monthly_emi,
			proposed_monthly_emi=request.proposed_monthly_emi,
			confirmed=request.confirmed,
			confirmation_token=request.confirmation_token,
		)
	except ValueError as error:
		raise HTTPException(status_code=422, detail=str(error)) from error

	if result["status"] == "confirmation_required":
		status_code = 202
	elif result["status"] == "ineligible":
		status_code = 422
	elif result["status"] == "submitted":
		status_code = 201
	else:
		status_code = 200
	return JSONResponse(status_code=status_code, content=result)


@app.post("/agent/chat")
def agent_chat(
	request: AgentChatRequest,
	authenticated_customer_id: str | None = Header(
		default=None,
		alias="X-Authenticated-Customer-ID",
	),
) -> dict[str, object]:
	"""Run one turn of the single tool-calling agent for an authenticated demo user."""
	if authenticated_customer_id is None:
		raise HTTPException(
			status_code=401,
			detail="Authentication is required to use the banking agent.",
		)
	try:
		return run_agent_turn(
			message=request.message,
			authenticated_customer_id=authenticated_customer_id,
			conversation_id=request.conversation_id,
		)
	except AgentAuthorizationError as error:
		raise HTTPException(status_code=403, detail=str(error)) from error
	except AgentConfigurationError as error:
		raise HTTPException(status_code=503, detail=str(error)) from error
	except AgentProviderError as error:
		raise HTTPException(status_code=502, detail=str(error)) from error
	except LookupError as error:
		raise HTTPException(status_code=404, detail=str(error)) from error