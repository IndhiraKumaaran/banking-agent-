"""Tests for the EMI calculator."""

import pytest

from app.mock_data import (
    MOCK_LOAN_APPLICATIONS,
    MOCK_LOAN_APPLICATIONS_BY_FINGERPRINT,
    PENDING_LOAN_CONFIRMATIONS,
)
from app.tools import (
    calculate_emi,
    check_loan_eligibility,
    get_account_details,
    submit_loan_application,
)


@pytest.fixture(autouse=True)
def clear_mock_loan_applications():
    MOCK_LOAN_APPLICATIONS.clear()
    MOCK_LOAN_APPLICATIONS_BY_FINGERPRINT.clear()
    PENDING_LOAN_CONFIRMATIONS.clear()


def test_calculate_emi_default_demonstration():
    result = calculate_emi(500_000, 12, 36)

    assert result == {
        "monthly_emi": 16607.15,
        "total_repayment": 597857.58,
        "total_interest": 97857.58,
    }


def test_calculate_emi_with_zero_interest():
    result = calculate_emi(120_000, 0, 12)

    assert result == {
        "monthly_emi": 10000.00,
        "total_repayment": 120000.00,
        "total_interest": 0.00,
    }


@pytest.mark.parametrize("loan_amount", [0, -1])
def test_calculate_emi_rejects_invalid_loan_amount(loan_amount):
    with pytest.raises(ValueError, match="Loan amount must be"):
        calculate_emi(loan_amount, 12, 36)


def test_calculate_emi_rejects_negative_interest_rate():
    with pytest.raises(ValueError, match="Annual interest rate must be"):
        calculate_emi(100_000, -1, 12)


@pytest.mark.parametrize("tenure_months", [0, -1])
def test_calculate_emi_rejects_nonpositive_tenure(tenure_months):
    with pytest.raises(ValueError, match="Loan tenure must be"):
        calculate_emi(100_000, 12, tenure_months)


def test_calculate_emi_rejects_noninteger_tenure():
    with pytest.raises(ValueError, match="Loan tenure must be"):
        calculate_emi(100_000, 12, 12.5)


def test_check_loan_eligibility_for_eligible_customer():
    result = check_loan_eligibility(60_000, 5_000, 20_000)

    assert result["eligible"] is True
    assert result["total_monthly_emi"] == 25_000.00
    assert result["maximum_permitted_emi"] == 30_000.00
    assert result["existing_emi_ratio"] == 0.0833
    assert result["total_emi_ratio"] == 0.4167


def test_check_loan_eligibility_for_sample_customer():
    proposed_emi = calculate_emi(500_000, 12, 36)["monthly_emi"]

    result = check_loan_eligibility(60_000, 45_000, proposed_emi, 0.50)

    assert result["eligible"] is False
    assert result["existing_emi_ratio"] == 0.75
    assert result["maximum_permitted_emi"] == 30_000.00
    assert result["total_monthly_emi"] == 61_607.15
    assert result["reason"] == (
        "Existing EMI obligations already exceed the illustrative permitted EMI "
        "limit, even before adding the proposed loan."
    )


def test_check_loan_eligibility_when_total_emi_exceeds_limit():
    result = check_loan_eligibility(60_000, 10_000, 21_000)

    assert result["eligible"] is False
    assert result["reason"] == "Total monthly EMI exceeds the illustrative permitted EMI limit."


def test_check_loan_eligibility_at_permitted_limit():
    result = check_loan_eligibility(60_000, 10_000, 20_000)

    assert result["eligible"] is True
    assert result["total_monthly_emi"] == result["maximum_permitted_emi"]


@pytest.mark.parametrize("monthly_salary", [0, -1])
def test_check_loan_eligibility_rejects_invalid_salary(monthly_salary):
    with pytest.raises(ValueError, match="Monthly salary must be"):
        check_loan_eligibility(monthly_salary, 0, 0)


def test_check_loan_eligibility_rejects_negative_existing_emi():
    with pytest.raises(ValueError, match="Existing monthly EMI must be"):
        check_loan_eligibility(60_000, -1, 0)


def test_check_loan_eligibility_rejects_negative_proposed_emi():
    with pytest.raises(ValueError, match="Proposed monthly EMI must be"):
        check_loan_eligibility(60_000, 0, -1)


@pytest.mark.parametrize("max_emi_ratio", [-0.1, 0, 1.01])
def test_check_loan_eligibility_rejects_invalid_emi_ratio(max_emi_ratio):
    with pytest.raises(ValueError, match="Maximum EMI ratio must be"):
        check_loan_eligibility(60_000, 0, 0, max_emi_ratio)


@pytest.mark.parametrize(
    ("input_name", "invalid_value"),
    [
        ("monthly_salary", float("nan")),
        ("monthly_salary", float("inf")),
        ("existing_monthly_emi", float("nan")),
        ("existing_monthly_emi", float("inf")),
        ("proposed_monthly_emi", float("nan")),
        ("proposed_monthly_emi", float("inf")),
        ("max_emi_ratio", float("nan")),
        ("max_emi_ratio", float("inf")),
    ],
)
def test_check_loan_eligibility_rejects_nonfinite_inputs(input_name, invalid_value):
    inputs = {
        "monthly_salary": 60_000,
        "existing_monthly_emi": 0,
        "proposed_monthly_emi": 0,
        "max_emi_ratio": 0.50,
    }
    inputs[input_name] = invalid_value

    with pytest.raises(ValueError):
        check_loan_eligibility(**inputs)


def test_get_account_details_for_authenticated_customer():
    account = get_account_details("CUST1001", "CUST1001")

    assert account == {
        "customer_id": "CUST1001",
        "customer_name": "Mira Example",
        "account_number": "XXXX1234",
        "account_type": "Savings",
        "balance": 84520.75,
    }


def test_get_account_details_rejects_missing_authentication():
    with pytest.raises(PermissionError, match="Authentication is required"):
        get_account_details("CUST1001", None)


def test_get_account_details_rejects_another_customer():
    with pytest.raises(PermissionError, match="not authorized"):
        get_account_details("CUST1002", "CUST1001")


def test_get_account_details_reports_unknown_customer():
    with pytest.raises(LookupError, match="CUST9999 was not found"):
        get_account_details("CUST9999", "CUST9999")


@pytest.mark.parametrize("customer_id", ["CUST1001", "CUST1002", "CUST1003"])
def test_get_account_details_returns_only_permitted_masked_fields(customer_id):
    account = get_account_details(customer_id, customer_id)

    assert account["account_number"].startswith("XXXX")
    assert len(account["account_number"]) == 8
    assert set(account) == {
        "customer_id",
        "customer_name",
        "account_number",
        "account_type",
        "balance",
    }
    assert not {"password", "pin", "cvv", "authentication_secret"} & set(account)


def sample_loan_application():
    return {
        "customer_id": "CUST1001",
        "loan_amount": 100_000,
        "tenure_months": 12,
        "annual_interest_rate": 0,
        "monthly_salary": 60_000,
        "existing_monthly_emi": 0,
        "proposed_monthly_emi": 8_333.33,
    }


def test_loan_application_requires_confirmation_without_storing_record():
    result = submit_loan_application(**sample_loan_application())

    assert result["status"] == "confirmation_required"
    assert result["application_summary"]["loan_amount"] == 100_000.0
    assert result["application_summary"]["tenure_months"] == 12
    assert result["confirmation_token"]
    assert MOCK_LOAN_APPLICATIONS == []


def test_confirmed_eligible_loan_application_is_stored():
    application_details = sample_loan_application()
    confirmation = submit_loan_application(**application_details)

    result = submit_loan_application(
        **application_details,
        confirmed=True,
        confirmation_token=confirmation["confirmation_token"],
    )

    assert result["status"] == "submitted"
    assert result["application"]["application_id"] == "MOCK-LOAN-000001"
    assert len(MOCK_LOAN_APPLICATIONS) == 1


def test_ineligible_loan_application_is_not_stored():
    application_details = sample_loan_application()
    application_details["existing_monthly_emi"] = 45_000

    result = submit_loan_application(**application_details, confirmed=True)

    assert result["status"] == "ineligible"
    assert "Existing EMI obligations" in result["message"]
    assert MOCK_LOAN_APPLICATIONS == []


def test_changed_loan_details_require_confirmation_again():
    initial_details = sample_loan_application()
    confirmation = submit_loan_application(**initial_details)
    changed_details = {
        **initial_details,
        "loan_amount": 120_000,
        "proposed_monthly_emi": 10_000,
    }

    refreshed_confirmation = submit_loan_application(
        **changed_details,
        confirmed=True,
        confirmation_token=confirmation["confirmation_token"],
    )

    assert refreshed_confirmation["status"] == "confirmation_required"
    assert refreshed_confirmation["confirmation_token"] != confirmation["confirmation_token"]
    assert refreshed_confirmation["application_summary"]["loan_amount"] == 120_000.0
    assert MOCK_LOAN_APPLICATIONS == []


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("loan_amount", 0),
        ("tenure_months", 0),
        ("annual_interest_rate", -1),
        ("monthly_salary", 0),
        ("existing_monthly_emi", -1),
        ("proposed_monthly_emi", -1),
    ],
)
def test_loan_application_rejects_invalid_inputs(field, invalid_value):
    application_details = sample_loan_application()
    application_details[field] = invalid_value

    with pytest.raises(ValueError):
        submit_loan_application(**application_details)


def test_loan_application_rejects_proposed_emi_not_matching_loan_terms():
    application_details = sample_loan_application()
    application_details["proposed_monthly_emi"] = 9_000

    with pytest.raises(ValueError, match="must match the EMI calculated"):
        submit_loan_application(**application_details)


def test_retried_confirmed_loan_application_is_not_duplicated():
    application_details = sample_loan_application()
    confirmation = submit_loan_application(**application_details)
    confirmation_details = {
        **application_details,
        "confirmed": True,
        "confirmation_token": confirmation["confirmation_token"],
    }

    first_result = submit_loan_application(**confirmation_details)
    retry_result = submit_loan_application(**confirmation_details)

    assert first_result["status"] == "submitted"
    assert retry_result["status"] == "already_submitted"
    assert retry_result["application"]["application_id"] == first_result["application"]["application_id"]
    assert len(MOCK_LOAN_APPLICATIONS) == 1