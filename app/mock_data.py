"""Fictional in-memory account records for local learning and tests."""

from typing import TypedDict


class CustomerAccount(TypedDict):
	"""Permitted account details returned by the mock lookup tool."""

	customer_id: str
	customer_name: str
	account_number: str
	account_type: str
	balance: float


class LoanApplicationRecord(TypedDict):
	"""Stored fields for a fictional submitted loan application."""

	application_id: str
	customer_id: str
	loan_amount: float
	tenure_months: int
	annual_interest_rate: float
	monthly_salary: float
	existing_monthly_emi: float
	proposed_monthly_emi: float
	status: str


CUSTOMER_ACCOUNTS: dict[str, CustomerAccount] = {
	"CUST1001": {
		"customer_id": "CUST1001",
		"customer_name": "Mira Example",
		"account_number": "XXXX1234",
		"account_type": "Savings",
		"balance": 84520.75,
	},
	"CUST1002": {
		"customer_id": "CUST1002",
		"customer_name": "Arjun Sample",
		"account_number": "XXXX5678",
		"account_type": "Current",
		"balance": 132450.00,
	},
	"CUST1003": {
		"customer_id": "CUST1003",
		"customer_name": "Leela Fiction",
		"account_number": "XXXX9012",
		"account_type": "Savings",
		"balance": 27680.50,
	},
}

MOCK_LOAN_APPLICATIONS: list[LoanApplicationRecord] = []
MOCK_LOAN_APPLICATIONS_BY_FINGERPRINT: dict[str, LoanApplicationRecord] = {}
PENDING_LOAN_CONFIRMATIONS: dict[str, str] = {}