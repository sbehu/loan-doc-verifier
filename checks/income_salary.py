"""
Salaried income verification check.
Three-way reconciliation: declared monthly income (application form),
net/take-home salary (salary slip), and actual salary credits (bank
statement, rows described as "Salary Credit - <employer>"). A salary
slip alone can be forged; cross-checking it against real deposits in
the bank statement is what makes this closer to how a real underwriter
verifies salaried income.

Returns the raw figures and their pairwise gap percentages rather than
a flagged bool -- the orchestrator applies the borderline band to
decide clean / investigate / reject, same principle as the UPI income
check and the signature check.
"""

import base64
import json
from pathlib import Path
from io import BytesIO

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from checks.textract_bank import extract_bank_statement_rows

base_dir = Path(__file__).resolve().parent.parent
load_dotenv(base_dir / ".env")
client = OpenAI()

CHUNK_HEIGHT = 900
OVERLAP = 150

DECLARED_INCOME_PROMPT = (
    "This is a loan application form. Find the 'Declared Monthly Income' "
    "field and the 'Employment Type' field. Extract the income as just the "
    "number, no currency symbol or commas. Normalize the employment type to "
    "exactly \"salaried\" or \"self_employed\" regardless of the exact "
    "wording shown. Return ONLY JSON: {\"declared_income\": <number>, "
    "\"employment_type\": \"salaried\" or \"self_employed\"}"
)

NET_SALARY_PROMPT = (
    "This is a salary slip / payslip. Layouts vary -- find whichever "
    "figure represents the final net/take-home pay after deductions "
    "(it may be labeled 'Net Salary Payable', 'Take Home', 'Net Pay', "
    "or similar -- not the gross figure). Extract just the number, no "
    "currency symbol or commas. Return ONLY JSON: {\"net_salary\": "
    "<number>}"
)

BANK_ROW_PROMPT = (
    "This is a horizontal strip cropped from a bank statement. Ignore "
    "any account holder name, address, account number, or bank branding "
    "if visible in this strip -- do not extract those. Only extract data "
    "from the transaction table rows. For every FULLY visible transaction "
    "row, extract: date, description, amount (positive for a deposit, "
    "negative for a withdrawal, no currency symbol or commas). Ignore "
    "any row visibly cut off at the top or bottom edge of this strip. "
    "Return ONLY JSON: {\"rows\": [{\"date\": ..., \"description\": ..., "
    "\"amount\": ...}]}"
)


def _encode(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def get_declared_income_and_employment(persona_id: str):
    """Returns (declared_income, employment_type). Reading employment type
    from the same call (rather than a separate one) means checking
    applicability doesn't cost an extra API call -- and applicability is
    decided by this declared value, not merely by which document files
    happen to exist (a persona can have a stray salary_slip.png present
    despite being declared self-employed)."""
    form_path = base_dir / "data" / "documents" / persona_id / "application_form.png"
    form = Image.open(form_path).convert("RGB")

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": DECLARED_INCOME_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode(form)}"}},
        ]}],
    )
    result = json.loads(resp.choices[0].message.content)
    return float(result["declared_income"]), result["employment_type"]


def get_net_salary(persona_id: str) -> float:
    slip_path = base_dir / "data" / "documents" / persona_id / "salary_slip.png"
    slip = Image.open(slip_path).convert("RGB")

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": NET_SALARY_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode(slip)}"}},
        ]}],
    )
    return float(json.loads(resp.choices[0].message.content)["net_salary"])


def get_bank_salary_credit(persona_id: str):
    """Returns (average, salary_rows) -- salary_rows is the raw list of
    individual deposits, exposed so the agent's investigation step can
    reason over the actual pattern, not just the averaged number.
    Returns (None, []) if no bank statement or no salary rows found.

    Extraction now goes through Textract's table detection (see
    textract_bank.py) rather than vision-LLM chunking -- same reasoning
    as the balance check, since bank_statement.png is a genuine table."""
    statement_path = base_dir / "data" / "documents" / persona_id / "bank_statement.png"
    if not statement_path.exists():
        return None, []

    rows = extract_bank_statement_rows(persona_id)
    salary_rows = [r for r in rows if "salary" in (r.get("description") or "").lower()]
    if not salary_rows:
        return None, []

    amounts = [float(r["amount"]) for r in salary_rows]
    return sum(amounts) / len(amounts), salary_rows

def check_income_salary(persona_id: str) -> dict:
    slip_path = base_dir / "data" / "documents" / persona_id / "salary_slip.png"
    if not slip_path.exists():
        return {
            "persona_id": persona_id,
            "check": "income_verification_salary",
            "applicable": False,
            "detail": "no salary slip present -- check not applicable (self-employed case)",
        }

    declared, employment_type = get_declared_income_and_employment(persona_id)
    if employment_type != "salaried":
        return {
            "persona_id": persona_id,
            "check": "income_verification_salary",
            "applicable": False,
            "detail": (
                f"applicant's declared employment type is '{employment_type}', not "
                "salaried -- salary check not applicable despite a salary slip file "
                "being present"
            ),
        }

    net_salary = get_net_salary(persona_id)
    bank_credit, salary_rows = get_bank_salary_credit(persona_id)

    def gap(a, b):
        if a is None or b is None:
            return None
        return abs(a - b) / a if a else 1.0

    gaps = {
        "declared_vs_slip": gap(declared, net_salary),
        "declared_vs_bank": gap(declared, bank_credit),
        "slip_vs_bank": gap(net_salary, bank_credit),
    }

    return {
        "persona_id": persona_id,
        "check": "income_verification_salary",
        "applicable": True,
        "declared": declared,
        "net_salary": net_salary,
        "bank_credit": bank_credit,
        "gaps": gaps,
        "detail": (
            f"declared {declared:.2f}, salary slip {net_salary:.2f}, "
            f"bank credits {bank_credit if bank_credit is not None else 'not found'}"
        ),
    }


if __name__ == "__main__":
    result = check_income_salary("P006_income_mismatch")
    print(result)
