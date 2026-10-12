"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.agent import AGENT_CONVERSATIONS, AgentProviderError
from app.mock_data import (
	MOCK_LOAN_APPLICATIONS,
	MOCK_LOAN_APPLICATIONS_BY_FINGERPRINT,
	PENDING_LOAN_CONFIRMATIONS,
)
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_mock_loan_applications():
	MOCK_LOAN_APPLICATIONS.clear()
	MOCK_LOAN_APPLICATIONS_BY_FINGERPRINT.clear()
	PENDING_LOAN_CONFIRMATIONS.clear()
	AGENT_CONVERSATIONS.clear()


def test_health_endpoint_returns_ok():
	response = client.get("/health")

	assert response.status_code == 200
	assert response.json() == {"status": "ok"}


def test_account_endpoint_returns_details_for_authenticated_customer():
	response = client.get(
		"/accounts/CUST1001",
		headers={"X-Authenticated-Customer-ID": "CUST1001"},
	)

	assert response.status_code == 200
	assert response.json()["customer_id"] == "CUST1001"
	assert response.json()["account_number"] == "XXXX1234"


def test_account_endpoint_rejects_missing_authentication():
	response = client.get("/accounts/CUST1001")

	assert response.status_code == 401
	assert response.json() == {
		"detail": "Authentication is required to view account details."
	}


def test_account_endpoint_rejects_access_to_another_customer():
	response = client.get(
		"/accounts/CUST1002",
		headers={"X-Authenticated-Customer-ID": "CUST1001"},
	)

	assert response.status_code == 403
	assert "not authorized" in response.json()["detail"]


def test_account_endpoint_returns_not_found_for_unknown_customer():
	response = client.get(
		"/accounts/CUST9999",
		headers={"X-Authenticated-Customer-ID": "CUST9999"},
	)

	assert response.status_code == 404
	assert "CUST9999 was not found" in response.json()["detail"]


def test_eligibility_endpoint_returns_result():
	response = client.post(
		"/loan/eligibility",
		json={
			"monthly_salary": 60_000,
			"existing_monthly_emi": 5_000,
			"proposed_monthly_emi": 20_000,
			"max_emi_ratio": 0.50,
		},
	)

	assert response.status_code == 200
	assert response.json()["eligible"] is True
	assert response.json()["total_monthly_emi"] == 25_000
	assert response.json()["maximum_permitted_emi"] == 30_000


@pytest.mark.parametrize(
	"invalid_values",
	[
		{"monthly_salary": 0},
		{"existing_monthly_emi": -1},
		{"max_emi_ratio": 1.1},
	],
)
def test_eligibility_endpoint_rejects_invalid_inputs(invalid_values):
	request_data = {
		"monthly_salary": 60_000,
		"existing_monthly_emi": 0,
		"proposed_monthly_emi": 0,
		"max_emi_ratio": 0.50,
	}
	request_data.update(invalid_values)

	response = client.post("/loan/eligibility", json=request_data)

	assert response.status_code == 422
	assert "detail" in response.json()


def sample_application_request():
	return {
		"customer_id": "CUST1001",
		"loan_amount": 100_000,
		"tenure_months": 12,
		"annual_interest_rate": 0,
		"monthly_salary": 60_000,
		"existing_monthly_emi": 0,
		"proposed_monthly_emi": 8_333.33,
	}


def test_loan_application_endpoint_returns_confirmation_summary_first():
	response = client.post("/loan/applications", json=sample_application_request())

	assert response.status_code == 202
	assert response.json()["status"] == "confirmation_required"
	assert response.json()["application_summary"]["loan_amount"] == 100_000
	assert response.json()["confirmation_token"]
	assert MOCK_LOAN_APPLICATIONS == []


def test_loan_application_endpoint_submits_after_matching_confirmation():
	request_data = sample_application_request()
	confirmation = client.post("/loan/applications", json=request_data).json()
	request_data.update(
		{
			"confirmed": True,
			"confirmation_token": confirmation["confirmation_token"],
		}
	)

	response = client.post("/loan/applications", json=request_data)

	assert response.status_code == 201
	assert response.json()["status"] == "submitted"
	assert len(MOCK_LOAN_APPLICATIONS) == 1


def test_loan_application_endpoint_requires_new_confirmation_for_changed_details():
	request_data = sample_application_request()
	confirmation = client.post("/loan/applications", json=request_data).json()
	request_data.update(
		{
			"loan_amount": 120_000,
			"proposed_monthly_emi": 10_000,
			"confirmed": True,
			"confirmation_token": confirmation["confirmation_token"],
		}
	)

	response = client.post("/loan/applications", json=request_data)

	assert response.status_code == 202
	assert response.json()["status"] == "confirmation_required"
	assert MOCK_LOAN_APPLICATIONS == []


def test_loan_application_endpoint_rejects_invalid_request_without_echoing_value():
	request_data = sample_application_request()
	request_data["loan_amount"] = -100_000

	response = client.post("/loan/applications", json=request_data)

	assert response.status_code == 422
	assert "input" not in response.text
	assert MOCK_LOAN_APPLICATIONS == []


def test_agent_chat_requires_authenticated_demo_identity():
	response = client.post("/agent/chat", json={"message": "Hello"})

	assert response.status_code == 401
	assert "Authentication is required" in response.json()["detail"]


def test_agent_chat_returns_clear_error_when_google_key_is_missing(monkeypatch):
	monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
	response = client.post(
		"/agent/chat",
		headers={"X-Authenticated-Customer-ID": "CUST1001"},
		json={"message": "Hello"},
	)

	assert response.status_code == 503
	assert response.json() == {
		"detail": "The agent is not configured. Set GOOGLE_API_KEY to enable chat."
	}


def test_agent_chat_returns_sanitized_provider_failure(monkeypatch):
	def fail_agent_turn(**kwargs):
		raise AgentProviderError("The chat model provider failed. Please try again.")

	monkeypatch.setattr(main_module, "run_agent_turn", fail_agent_turn)
	response = client.post(
		"/agent/chat",
		headers={"X-Authenticated-Customer-ID": "CUST1001"},
		json={"message": "Hello"},
	)

	assert response.status_code == 502
	assert response.json() == {
		"detail": "The chat model provider failed. Please try again."
	}