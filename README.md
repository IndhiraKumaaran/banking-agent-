# Smart Banking Support Agent Using Agentic AI

This repository is an educational demo for a small banking-support workflow. It combines an EMI calculator, an illustrative affordability checker, fictional mock account data, a FastAPI service, and a single LangChain tool-calling agent. It does not connect to a real bank or persist data beyond the in-memory demo state.

## Architecture overview

The project is intentionally small and deterministic:

- `app/tools.py` contains the production-style demo logic for EMI calculation, affordability checks, mock account access, and loan-application confirmation.
- `app/agent.py` builds a single LangChain agent with five tools. The agent enforces a fixed order of operations and uses exact confirmation logic in Python rather than relying on model prompts alone.
- `app/main.py` exposes the demo through FastAPI endpoints and converts Python errors into HTTP responses.
- `app/mock_data.py` contains the in-memory customer records and application state used by the mock banking workflow.
- `tests/` verifies the business rules, API behavior, and the agent's verification steps.

The design is meant to demonstrate agent orchestration and guardrails, not a production banking stack.

## Setup and environment

### Prerequisites

- Python 3.11+ recommended
- A virtual environment
- Optional: a Google Gemini API key for the `/agent/chat` endpoint

### Install dependencies

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Configure Gemini access

The non-agent logic and tests do not require an LLM API key. The `/agent/chat` route uses Google Gemini and expects `GOOGLE_API_KEY` in the project-root `.env` file or in the environment.

The application loads `.env` from the project root automatically before creating the model. `GEMINI_MODEL` is optional; if unset, the code defaults to `gemini-2.5-flash`.

The repository includes an `.env.example` template for local setup. Do not commit a real key or paste secret material into the README.

Example pattern:

```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY to a local placeholder or development value.
```

If `GOOGLE_API_KEY` is missing or still set to the example placeholder value, the agent returns a clear configuration error instead of attempting a provider call.

## Authentication and mock data model

The product uses fictional in-memory records only:

- `customer_id` values such as `CUST1001`, `CUST1002`, and `CUST1003` are demo identities.
- A request is considered authenticated only when the caller supplies the matching `X-Authenticated-Customer-ID` header.
- Account lookup is bound to that customer identity and forbids access to another customer record.
- Returned account numbers are masked as `XXXX####` and do not expose full account numbers, passwords, PINs, OTPs, CVVs, or other sensitive values.
- The app stores application state in memory only; restarting the process clears it.

This is explicitly a learning simulation. It is not a secure authentication system and not a production bank implementation.

## Run the API locally

From the project root, start the FastAPI app:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Check that the service is alive:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

Open the interactive API docs at `/docs` once the app is running.

## API reference

### `GET /health`

Health check for the service.

Example:

```bash
curl http://localhost:8000/health
```

Response:

```json
{"status":"ok"}
```

### `GET /accounts/{customer_id}`

Returns masked mock account data for the authenticated customer.

Required header:

```http
X-Authenticated-Customer-ID: CUST1001
```

Example:

```bash
curl -H "X-Authenticated-Customer-ID: CUST1001" \
  http://localhost:8000/accounts/CUST1001
```

Example response:

```json
{
  "customer_id": "CUST1001",
  "customer_name": "Mira Example",
  "account_number": "XXXX1234",
  "account_type": "Savings",
  "balance": 84520.75
}
```

Behavior:

- Missing auth header -> `401`
- Different authenticated customer than requested customer -> `403`
- Unknown customer ID -> `404`

### `POST /loan/eligibility`

Evaluates a simplified EMI affordability check for a loan request.

Request body:

```json
{
  "monthly_salary": 60000,
  "existing_monthly_emi": 5000,
  "proposed_monthly_emi": 20000,
  "max_emi_ratio": 0.5
}
```

Response example:

```json
{
  "eligible": true,
  "monthly_salary": 60000.0,
  "existing_monthly_emi": 5000.0,
  "proposed_monthly_emi": 20000.0,
  "total_monthly_emi": 25000.0,
  "maximum_permitted_emi": 30000.0,
  "existing_emi_ratio": 0.0833,
  "total_emi_ratio": 0.4167,
  "reason": "Total monthly EMI is within the illustrative permitted EMI limit."
}
```

Validation rules:

- `monthly_salary` must be a finite number greater than zero
- `existing_monthly_emi` must be zero or greater
- `proposed_monthly_emi` must be zero or greater
- `max_emi_ratio` must be greater than zero and no greater than one
- invalid values return an HTTP `422` with a validation error payload

### `POST /loan/applications`

Prepares or submits a mock loan application. This endpoint uses deterministic Python validation and confirmation logic.

Request body pattern:

```json
{
  "customer_id": "CUST1001",
  "loan_amount": 100000,
  "tenure_months": 12,
  "annual_interest_rate": 0,
  "monthly_salary": 60000,
  "existing_monthly_emi": 0,
  "proposed_monthly_emi": 8333.33,
  "confirmed": false,
  "confirmation_token": null
}
```

Behavior by state:

- If the application is not yet confirmed, it returns `202` with `status: "confirmation_required"` and a `confirmation_token`.
- If the request is invalid, it returns `422`.
- If the application is ineligible, it returns `422` with `status: "ineligible"`.
- If a valid, confirmed request matches the summary, it returns `201` with `status: "submitted"`.
- If the same confirmed payload is resent, it returns the existing mock application instead of creating a duplicate entry.

Example first step (not yet confirmed):

```bash
curl -X POST http://localhost:8000/loan/applications \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id":"CUST1001",
    "loan_amount":100000,
    "tenure_months":12,
    "annual_interest_rate":0,
    "monthly_salary":60000,
    "existing_monthly_emi":0,
    "proposed_monthly_emi":8333.33
  }'
```

Example response shape:

```json
{
  "status": "confirmation_required",
  "message": "Review the application summary, then confirm these same details using the returned confirmation token.",
  "application_summary": {
    "customer_id": "CUST1001",
    "loan_amount": 100000.0,
    "tenure_months": 12,
    "annual_interest_rate": 0.0,
    "monthly_emi": 8333.33,
    "monthly_salary": 60000.0,
    "existing_monthly_emi": 0.0,
    "proposed_monthly_emi": 8333.33,
    "max_emi_ratio": 0.5,
    "total_monthly_emi": 8333.33,
    "eligible": true
  },
  "confirmation_token": "REDACTED_TOKEN"
}
```

When the summary is confirmed, the client must send the same values together with the returned token:

```json
{
  "customer_id": "CUST1001",
  "loan_amount": 100000,
  "tenure_months": 12,
  "annual_interest_rate": 0,
  "monthly_salary": 60000,
  "existing_monthly_emi": 0,
  "proposed_monthly_emi": 8333.33,
  "confirmed": true,
  "confirmation_token": "REDACTED_TOKEN"
}
```

If any detail changes, the confirmation token no longer matches and the API returns a new confirmation-required response.

### `POST /agent/chat`

Runs a single guarded LangChain agent turn for an authenticated customer.

Required header:

```http
X-Authenticated-Customer-ID: CUST1001
```

Request body:

```json
{
  "message": "I want a personal loan of Rs. 100000 at 12% for 12 months. My monthly salary is Rs. 60000 and my existing monthly EMI is Rs. 0.",
  "conversation_id": "optional-existing-conversation-id"
}
```

The agent enforces the following flow:

1. It uses the customer identity provided by the HTTP header.
2. It binds each conversation to a single authenticated customer.
3. It calls `calculate_emi` before checking affordability.
4. It calls `check_loan_eligibility` only with the exact EMI just calculated.
5. It prepares the loan application summary and, if eligible, asks for an exact confirmation phrase.
6. It only submits an application after the user replies with the exact phrase `I confirm the loan application details exactly as shown.`
7. It rejects vague replies such as `yes`.

Example:

```bash
curl -X POST http://localhost:8000/agent/chat \
  -H "X-Authenticated-Customer-ID: CUST1001" \
  -H "Content-Type: application/json" \
  -d '{"message":"I want a personal loan of Rs. 100000 at 12% for 12 months. My monthly salary is Rs. 60000 and my existing monthly EMI is Rs. 0."}'
```

The response includes a `conversation_id` for follow-up turns. To confirm the pending application, send the same `conversation_id` and the exact confirmation text in a second request.

Example confirmation request:

```bash
curl -X POST http://localhost:8000/agent/chat \
  -H "X-Authenticated-Customer-ID: CUST1001" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id":"DEMO_CONVERSATION_ID",
    "message":"I confirm the loan application details exactly as shown."
  }'
```

If the model is misconfigured or the provider fails, the API returns sanitized errors. Missing `GOOGLE_API_KEY` returns HTTP `503` with a clear configuration message; provider failures return `502` with a generic provider error.

## EMI and eligibility rules

The demo uses a standard amortized EMI formula:

- If the annual interest rate is zero, the EMI is simply `loan_amount / tenure_months`.
- Otherwise, the monthly rate is `annual_interest_rate / 12 / 100` and the EMI is calculated as:

```text
monthly_emi = P * r * (1 + r)^n / ((1 + r)^n - 1)
```

Where:

- `P` = principal loan amount
- `r` = monthly interest rate
- `n` = number of monthly installments

The tool returns these rounded values:

- `monthly_emi`
- `total_repayment`
- `total_interest`

The affordability rule is intentionally simple:

```text
maximum_permitted_emi = monthly_salary * max_emi_ratio

total_monthly_emi = existing_monthly_emi + proposed_monthly_emi

eligible = total_monthly_emi <= maximum_permitted_emi
```

The default `max_emi_ratio` is `0.50`.

The reason strings returned by the tool are:

- `"Existing EMI obligations already exceed the illustrative permitted EMI limit, even before adding the proposed loan."`
- `"Total monthly EMI is within the illustrative permitted EMI limit."`
- `"Total monthly EMI exceeds the illustrative permitted EMI limit."`

## Confirmation and submission safeguards

The mock application flow is designed to prevent accidental confirmation.

- The application summary is serialized as a canonical JSON payload and hashed.
- The backend generates a `confirmation_token` that binds to the exact summary values.
- If the user changes any detail, the token no longer matches and a new summary is generated.
- The exact confirmation phrase is enforced in Python and is not delegated to the LLM prompt alone.
- Vague confirmations such as `yes` do not submit anything.
- Only eligible applications are allowed into the in-memory store.
- The mock application store is in memory only and resets when the process restarts.

This is intentionally a demonstration of guardrails, not a production identity or authorization system.

## Test instructions

Run the project tests from the repo root:

```bash
python -m pytest -v
```

The repository was verified locally with the following result:

```text
77 passed
```

## Limitations

This project intentionally has several limitations:

- It uses fictional, in-memory data only.
- It does not connect to a real bank or financial back end.
- It does not use persistent databases or durable sessions.
- It does not implement real user authentication, session management, or authorization.
- It does not provide compliance, regulatory, or fraud controls.
- It relies on a deterministic Python check for loan logic and confirmation, while the LLM is used to orchestrate the workflow.

This demo is for educational use, learning, and experimentation with agentic workflows, not production banking software.

## Summary

This project demonstrates a compact agentic workflow in which a language model calls deterministic tools to:

- calculate an EMI,
- evaluate affordability,
- fetch masked customer account information,
- prepare a mock application summary,
- and require explicit confirmation before a mock submission is stored.

The README focuses on the actual, verified behavior in the repository rather than making claims beyond the code and tests.
