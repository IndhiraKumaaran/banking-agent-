"""Reusable tools for banking support tasks."""

import math
import hashlib
import json
import secrets
from typing import Any

from app.mock_data import (
	CUSTOMER_ACCOUNTS,
	MOCK_LOAN_APPLICATIONS,
	MOCK_LOAN_APPLICATIONS_BY_FINGERPRINT,
	PENDING_LOAN_CONFIRMATIONS,
	CustomerAccount,
	LoanApplicationRecord,
)


def calculate_emi(
	loan_amount: float,
	annual_interest_rate: float,
	tenure_months: int,
) -> dict[str, float]:
	"""Calculate monthly EMI, total repayment, and total interest.

	Args:
		loan_amount: Principal loan amount, greater than zero.
		annual_interest_rate: Annual interest rate as a percentage, zero or greater.
		tenure_months: Loan tenure as a positive whole number of months.

	Returns:
		A dictionary with ``monthly_emi``, ``total_repayment``, and
		``total_interest`` amounts rounded to two decimal places.

	Raises:
		ValueError: If any input is outside its valid range or has an invalid type.
	"""
	if (
		isinstance(loan_amount, bool)
		or not isinstance(loan_amount, (int, float))
		or not math.isfinite(loan_amount)
		or loan_amount <= 0
	):
		raise ValueError("Loan amount must be a finite number greater than zero.")

	if (
		isinstance(annual_interest_rate, bool)
		or not isinstance(annual_interest_rate, (int, float))
		or not math.isfinite(annual_interest_rate)
		or annual_interest_rate < 0
	):
		raise ValueError("Annual interest rate must be a finite number zero or greater.")

	if isinstance(tenure_months, bool) or not isinstance(tenure_months, int):
		raise ValueError("Loan tenure must be a positive integer number of months.")
	if tenure_months <= 0:
		raise ValueError("Loan tenure must be a positive integer number of months.")

	if annual_interest_rate == 0:
		monthly_emi = loan_amount / tenure_months
	else:
		monthly_rate = annual_interest_rate / 12 / 100
		rate_factor = (1 + monthly_rate) ** tenure_months
		monthly_emi = loan_amount * monthly_rate * rate_factor / (rate_factor - 1)

	total_repayment = monthly_emi * tenure_months
	total_interest = total_repayment - loan_amount

	return {
		"monthly_emi": round(monthly_emi, 2),
		"total_repayment": round(total_repayment, 2),
		"total_interest": round(total_interest, 2),
	}


def check_loan_eligibility(
	monthly_salary: float,
	existing_monthly_emi: float,
	proposed_monthly_emi: float,
	max_emi_ratio: float = 0.50,
) -> dict[str, bool | float | str]:
	"""Check affordability using an illustrative maximum EMI-to-salary ratio.

	Args:
		monthly_salary: Customer's monthly salary, greater than zero.
		existing_monthly_emi: Current monthly EMI obligations, zero or greater.
		proposed_monthly_emi: New monthly EMI being considered, zero or greater.
		max_emi_ratio: Maximum share of salary allowed for total EMIs, in (0, 1].

	Returns:
		A dictionary with the eligibility decision, rounded amounts and ratios,
		and a plain-language reason.

	Raises:
		ValueError: If an input is non-numeric, non-finite, or outside its valid range.
	"""
	if (
		isinstance(monthly_salary, bool)
		or not isinstance(monthly_salary, (int, float))
		or not math.isfinite(monthly_salary)
		or monthly_salary <= 0
	):
		raise ValueError("Monthly salary must be a finite number greater than zero.")

	if (
		isinstance(existing_monthly_emi, bool)
		or not isinstance(existing_monthly_emi, (int, float))
		or not math.isfinite(existing_monthly_emi)
		or existing_monthly_emi < 0
	):
		raise ValueError("Existing monthly EMI must be a finite number zero or greater.")

	if (
		isinstance(proposed_monthly_emi, bool)
		or not isinstance(proposed_monthly_emi, (int, float))
		or not math.isfinite(proposed_monthly_emi)
		or proposed_monthly_emi < 0
	):
		raise ValueError("Proposed monthly EMI must be a finite number zero or greater.")

	if (
		isinstance(max_emi_ratio, bool)
		or not isinstance(max_emi_ratio, (int, float))
		or not math.isfinite(max_emi_ratio)
		or max_emi_ratio <= 0
		or max_emi_ratio > 1
	):
		raise ValueError("Maximum EMI ratio must be a finite number greater than zero and no greater than 1.")

	maximum_permitted_emi = monthly_salary * max_emi_ratio
	total_monthly_emi = existing_monthly_emi + proposed_monthly_emi
	eligible = total_monthly_emi <= maximum_permitted_emi

	if existing_monthly_emi > maximum_permitted_emi:
		reason = (
			"Existing EMI obligations already exceed the illustrative permitted EMI "
			"limit, even before adding the proposed loan."
		)
	elif eligible:
		reason = "Total monthly EMI is within the illustrative permitted EMI limit."
	else:
		reason = "Total monthly EMI exceeds the illustrative permitted EMI limit."

	return {
		"eligible": eligible,
		"monthly_salary": round(monthly_salary, 2),
		"existing_monthly_emi": round(existing_monthly_emi, 2),
		"proposed_monthly_emi": round(proposed_monthly_emi, 2),
		"total_monthly_emi": round(total_monthly_emi, 2),
		"maximum_permitted_emi": round(maximum_permitted_emi, 2),
		"existing_emi_ratio": round(existing_monthly_emi / monthly_salary, 4),
		"total_emi_ratio": round(total_monthly_emi / monthly_salary, 4),
		"reason": reason,
	}


def get_account_details(
	customer_id: str,
	authenticated_customer_id: str | None,
) -> CustomerAccount:
	"""Return masked account details after a simulated identity check.

	The authenticated customer ID is assumed to come from a trusted caller. This
	mock check is for learning only and is not production-grade authentication.

	Raises:
		PermissionError: If the caller is unauthenticated or requests another
			customer's account.
		LookupError: If the authenticated customer ID does not exist.
	"""
	if authenticated_customer_id is None:
		raise PermissionError("Authentication is required to view account details.")

	if authenticated_customer_id != customer_id:
		raise PermissionError("You are not authorized to view this customer's account.")

	if customer_id not in CUSTOMER_ACCOUNTS:
		raise LookupError(f"Customer ID {customer_id} was not found.")

	account = CUSTOMER_ACCOUNTS[customer_id]
	return {
		"customer_id": account["customer_id"],
		"customer_name": account["customer_name"],
		"account_number": account["account_number"],
		"account_type": account["account_type"],
		"balance": account["balance"],
	}


def submit_loan_application(
	customer_id: str,
	loan_amount: float,
	tenure_months: int,
	annual_interest_rate: float,
	monthly_salary: float,
	existing_monthly_emi: float,
	proposed_monthly_emi: float,
	confirmed: bool = False,
	confirmation_token: str | None = None,
) -> dict[str, Any]:
	"""Prepare or submit an eligible mock loan application.

	A confirmation token binds approval to all displayed application details. It
	prevents accidental confirmation of changed details; it is not authentication
	or a production security token. Repeated confirmed submissions are idempotent
	within this in-memory demo.
	"""
	if not isinstance(customer_id, str) or not customer_id.strip():
		raise ValueError("Customer ID must be a non-empty string.")
	if not isinstance(confirmed, bool):
		raise ValueError("Customer confirmation must be true or false.")

	loan_terms = calculate_emi(loan_amount, annual_interest_rate, tenure_months)
	calculated_monthly_emi = loan_terms["monthly_emi"]
	if (
		isinstance(proposed_monthly_emi, bool)
		or not isinstance(proposed_monthly_emi, (int, float))
		or not math.isfinite(proposed_monthly_emi)
		or proposed_monthly_emi < 0
	):
		raise ValueError("Proposed monthly EMI must be a finite number zero or greater.")
	if round(proposed_monthly_emi, 2) != calculated_monthly_emi:
		raise ValueError("Proposed monthly EMI must match the EMI calculated from the loan terms.")

	eligibility = check_loan_eligibility(
		monthly_salary=monthly_salary,
		existing_monthly_emi=existing_monthly_emi,
		proposed_monthly_emi=calculated_monthly_emi,
	)
	application_summary = {
		"customer_id": customer_id,
		"loan_amount": float(loan_amount),
		"tenure_months": tenure_months,
		"annual_interest_rate": float(annual_interest_rate),
		"monthly_emi": calculated_monthly_emi,
		"monthly_salary": float(monthly_salary),
		"existing_monthly_emi": float(existing_monthly_emi),
		"proposed_monthly_emi": calculated_monthly_emi,
		"max_emi_ratio": 0.50,
		"total_monthly_emi": eligibility["total_monthly_emi"],
		"eligible": eligibility["eligible"],
	}

	if not eligibility["eligible"]:
		return {
			"status": "ineligible",
			"message": eligibility["reason"],
			"application_summary": application_summary,
		}

	confirmation_payload = json.dumps(
		application_summary,
		sort_keys=True,
		separators=(",", ":"),
		allow_nan=False,
	)
	application_fingerprint = hashlib.sha256(
		confirmation_payload.encode("utf-8")
	).hexdigest()

	if (
		not confirmed
		or confirmation_token is None
		or PENDING_LOAN_CONFIRMATIONS.get(confirmation_token) != application_fingerprint
	):
		current_confirmation_token = secrets.token_urlsafe(32)
		PENDING_LOAN_CONFIRMATIONS[current_confirmation_token] = application_fingerprint
		return {
			"status": "confirmation_required",
			"message": (
				"Review the application summary, then confirm these same details "
				"using the returned confirmation token."
			),
			"application_summary": application_summary,
			"confirmation_token": current_confirmation_token,
		}

	existing_application = MOCK_LOAN_APPLICATIONS_BY_FINGERPRINT.get(application_fingerprint)
	if existing_application is not None:
		return {
			"status": "already_submitted",
			"message": "This confirmed application was already submitted.",
			"application": existing_application,
		}

	application: LoanApplicationRecord = {
		"application_id": f"MOCK-LOAN-{len(MOCK_LOAN_APPLICATIONS) + 1:06d}",
		"customer_id": customer_id,
		"loan_amount": float(loan_amount),
		"tenure_months": tenure_months,
		"annual_interest_rate": float(annual_interest_rate),
		"monthly_salary": float(monthly_salary),
		"existing_monthly_emi": float(existing_monthly_emi),
		"proposed_monthly_emi": calculated_monthly_emi,
		"status": "submitted",
	}
	MOCK_LOAN_APPLICATIONS.append(application)
	MOCK_LOAN_APPLICATIONS_BY_FINGERPRINT[application_fingerprint] = application
	return {
		"status": "submitted",
		"message": "Mock loan application submitted successfully.",
		"application": application,
	}