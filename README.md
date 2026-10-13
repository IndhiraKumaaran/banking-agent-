# Smart Banking Support Agent Using Agentic AI

This project is an incremental educational demo for common banking-support workflows. It includes an EMI calculator, an illustrative eligibility checker, fictional mock account data, a FastAPI interface, and one LangChain tool-calling agent. It does not connect to a real bank.

## Planned Development

1. Set up the project structure and development dependencies.
2. Add the EMI calculator, eligibility checker, and mock account lookup.
3. Expose the support tools through FastAPI endpoints.
4. Integrate the banking tools into one LangChain agent.
5. Expand and run tests for the project behavior.

## Install in GitHub Codespaces

Open a terminal in your Codespace, then run these commands from the project folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The non-agent tools and their tests do not require an LLM API key. The `/agent/chat` endpoint uses Google Gemini and requires `GOOGLE_API_KEY`.

## Configure the Agent

Copy `.env.example` to `.env` and add your Gemini API key, or configure `GOOGLE_API_KEY` as a Codespaces secret. The application loads the project-root `.env` file automatically. Do not commit a real key; `.env` is ignored by Git. The example file contains placeholders only.

```bash
cp .env.example .env
# Edit .env to set GOOGLE_API_KEY to your Gemini API key.
```

`GEMINI_MODEL` is optional; the default is `gemini-2.5-flash`, a Gemini model with LangChain tool-calling support. Set it to another currently available tool-capable Gemini model if you prefer. For automated tests, the agent uses a scripted fake model and does not make provider calls.

## Authentication and Mock Data

The account lookup uses fictional in-memory records and treats the authenticated customer ID as a simulated identity supplied by a trusted caller. This mock check is for training only. Real banking systems require proper identity verification, secure sessions or tokens, and server-side authorization.

## Run the Tests

With the virtual environment activated, run:

```bash
python -m pytest -v
```

## Start the API in GitHub Codespaces

With the dependencies installed and the virtual environment activated, first check whether the API is already running:

```bash
curl http://localhost:8000/health
```

If this returns `{"status":"ok"}`, reuse the existing server; do not start a second Uvicorn process on port 8000. If you need to restart a server in a terminal you control, stop it with **Ctrl+C** there, then start it again from the project folder:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

If the original terminal is unavailable, identify the listener before taking action:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
ps -fp <PID>
```

Only if the process is the Uvicorn server you intend to restart, request a graceful stop with `kill -TERM <PID>`. Do not stop an unfamiliar process. Then run the Uvicorn command above.

In Codespaces, open the **Ports** tab in the bottom panel. If port `8000` is not listed, choose **Forward a Port** and enter `8000`. Open the forwarded address for port 8000, then add `/docs` to that address to use the interactive API documentation. Forwarded ports are private by default; keep that setting unless you specifically need otherwise.

## Chat with the Agent

The agent endpoint requires the demo identity header. The header is supplied by the API caller in this training project; it is not secure production authentication. The agent uses this identity for account lookup and binds each in-memory conversation to that customer. Live Gemini requests are not required for tests and have not been verified as part of this project test run.

Start with a normal loan request. The agent calculates EMI, checks eligibility, and if eligible returns the exact details plus a `conversation_id` and confirmation token. For an ineligible request, it explains the result and does not ask for confirmation.

```bash
curl -X POST http://localhost:8000/agent/chat \
	-H "X-Authenticated-Customer-ID: CUST1001" \
	-H "Content-Type: application/json" \
	-d '{"message":"I want a personal loan of Rs. 100000 at 12% for 12 months. My monthly salary is Rs. 60000 and my existing monthly EMI is Rs. 0."}'
```

After reviewing the exact returned summary, confirm in the same conversation with the exact phrase shown by the agent:

```bash
curl -X POST http://localhost:8000/agent/chat \
	-H "X-Authenticated-Customer-ID: CUST1001" \
	-H "Content-Type: application/json" \
	-d '{"conversation_id":"YOUR_CONVERSATION_ID","message":"I confirm the loan application details exactly as shown."}'
```

The agent accepts no vague confirmation such as `yes`; the confirmation handler and existing submission tool enforce this in Python rather than relying on the prompt. Conversation state and applications are in memory and are lost when the process restarts.

### Sample Requests

Retrieve the fictional account for `CUST1001` by supplying the simulated authenticated customer ID:

```bash
curl -H "X-Authenticated-Customer-ID: CUST1001" \
	http://localhost:8000/accounts/CUST1001
```

Check illustrative loan eligibility:

```bash
curl -X POST http://localhost:8000/loan/eligibility \
	-H "Content-Type: application/json" \
	-d '{"monthly_salary": 60000, "existing_monthly_emi": 5000, "proposed_monthly_emi": 20000, "max_emi_ratio": 0.5}'
```

### Submit a Mock Loan Application

Send the application details with `confirmed` omitted or `false` to receive a summary and confirmation token. Review those exact details, then send the same request with `confirmed: true` and the returned `confirmation_token`. If any details change, the token no longer matches and the API returns a refreshed summary requiring confirmation again. These mock confirmation tokens are held in memory and are not a production authentication or security mechanism.

```bash
curl -X POST http://localhost:8000/loan/applications \
	-H "Content-Type: application/json" \
	-d '{"customer_id":"CUST1001","loan_amount":100000,"tenure_months":12,"annual_interest_rate":0,"monthly_salary":60000,"existing_monthly_emi":0,"proposed_monthly_emi":8333.33}'
```

Only eligible, explicitly confirmed applications are added to an in-memory mock store. Repeating an already confirmed request returns the existing mock application instead of creating another record. This educational flow is not production banking software and does not submit an application to a real bank.

The account header is only a training simulation, not secure authentication. Real banking systems require proper identity verification, secure sessions or tokens, and server-side authorization.

## Limitations

This prototype uses fictional data, an in-memory store, and a simulated identity header. LLM output can be incorrect, so deterministic Python functions enforce EMI calculations, eligibility, account authorization, and application confirmation. It has no durable storage, real bank connection, production authentication, or regulatory/compliance status; it is not production banking software.