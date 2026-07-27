"""
Income verification check (self-employed / UPI-based).
Not all UPI providers print a header summary (Google Pay does; PhonePe
and Paytm don't) so we can't rely on that. Instead we extract individual
transaction rows from overlapping strips (same pattern as the balance
check), sum only credit (positive) amounts as income -- excluding rows
explicitly tagged "(personal)" on the statement, since those are marked
by the provider as non-business transfers -- and derive the statement
period from the min/max dates actually seen in the rows. Average
monthly income = total credits / months in period, summed across all
UPI accounts for multi-account personas.

Returns the raw gap_pct as a number rather than a flagged bool -- the
orchestrator applies the borderline band (small gap = clean, moderate
gap = investigate further, large gap = flag) rather than this file
deciding it alone.
"""

import base64
import json
from datetime import date
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

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

INCOME_ROW_PROMPT = (
    "This is a horizontal strip cropped from a UPI transaction history "
    "(could be Google Pay, PhonePe, or Paytm format). Ignore any account "
    "holder name or phone number if visible -- do not extract those. "
    "Some rows are grouped under a date header (e.g. '18 April 2026') "
    "that applies to all rows below it until the next date header. If "
    "the date header for a row's group is visible in THIS strip, extract "
    "it as that row's date in YYYY-MM-DD format; if the group's date "
    "header is not visible in this strip (cut off above), return "
    "\"date\": null for that row -- do not guess. For every FULLY "
    "visible transaction row, extract: description (the 'Received from "
    "X' or 'Paid to X' text), amount (positive number for a "
    "credit/received row, negative for a debit/paid row, no currency "
    "symbol or commas), and date (or null). Ignore any row visibly cut "
    "off at the top or bottom edge of this strip. Return ONLY JSON: "
    "{\"rows\": [{\"description\": ..., \"amount\": ..., \"date\": ...}]}"
)


def _encode(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def get_declared_income_and_employment(persona_id: str):
    """Returns (declared_income, employment_type). Reading employment type
    from the same call avoids a separate API call, and lets applicability
    be decided by the declared value rather than merely by which document
    files happen to exist (a persona can have a stray upi_statement.png
    present despite being declared salaried, or vice versa)."""
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


def extract_rows_from_chunk(chunk: Image.Image, y_position: int, max_retries: int = 3) -> list:
    for attempt in range(1, max_retries + 1):
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": INCOME_ROW_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode(chunk)}"}},
            ]}],
        )
        content = resp.choices[0].message.content
        if content is not None:
            return json.loads(content)["rows"]
        print(f"DEBUG - chunk at y={y_position} refused (attempt {attempt}/{max_retries}): "
              f"{resp.choices[0].message.refusal}")
    print(f"WARNING - chunk at y={y_position} refused after {max_retries} attempts, skipping this chunk")
    return []


def extract_income_rows(statement_path: Path):
    """Returns (rows, dates_seen) for one statement -- exposed as its own
    function since this is exactly what the agent's "dig deeper" step
    will call when the income gap lands in the borderline zone, to look
    at the raw transaction rows instead of just the summary number."""
    img = Image.open(statement_path).convert("RGB")
    w, h = img.size

    seen = {}  # dedup key -> row
    dates_seen = []
    y = 0
    stride = CHUNK_HEIGHT - OVERLAP
    while y < h:
        chunk = img.crop((0, y, w, min(y + CHUNK_HEIGHT, h)))
        chunk_rows = extract_rows_from_chunk(chunk, y)
        for row in chunk_rows:
            try:
                amount = float(row.get("amount"))
            except (TypeError, ValueError):
                continue
            key = (row.get("description"), amount)
            seen[key] = row
            if row.get("date"):
                try:
                    dates_seen.append(date.fromisoformat(row["date"]))
                except ValueError:
                    pass
        y += stride

    return list(seen.values()), dates_seen


def get_actual_monthly_income(persona_id: str) -> float:
    doc_dir = base_dir / "data" / "documents" / persona_id
    statement_paths = sorted(doc_dir.glob("upi_statement*.png"))

    total_credit = 0.0
    all_dates = []
    for path in statement_paths:
        rows, dates_seen = extract_income_rows(path)
        credit_sum = sum(
            float(r["amount"]) for r in rows
            if float(r["amount"]) > 0 and "(personal)" not in (r.get("description") or "").lower()
        )
        total_credit += credit_sum
        all_dates.extend(dates_seen)

    if not all_dates:
        raise ValueError("no dates extracted from any UPI statement -- cannot determine period")

    months = max((max(all_dates) - min(all_dates)).days / 30, 1)
    return total_credit / months


def check_income_upi(persona_id: str) -> dict:
    doc_dir = base_dir / "data" / "documents" / persona_id
    if not list(doc_dir.glob("upi_statement*.png")):
        return {
            "persona_id": persona_id,
            "check": "income_verification_upi",
            "applicable": False,
            "detail": "no UPI statement present -- check not applicable (salaried case)",
        }

    declared, employment_type = get_declared_income_and_employment(persona_id)
    if employment_type != "self_employed":
        return {
            "persona_id": persona_id,
            "check": "income_verification_upi",
            "applicable": False,
            "detail": (
                f"applicant's declared employment type is '{employment_type}', not "
                "self-employed -- UPI check not applicable despite a UPI statement "
                "file being present"
            ),
        }

    actual = get_actual_monthly_income(persona_id)
    gap_pct = abs(declared - actual) / declared if declared else 1.0

    return {
        "persona_id": persona_id,
        "check": "income_verification_upi",
        "applicable": True,
        "declared": declared,
        "actual": actual,
        "gap_pct": gap_pct,
        "detail": f"declared {declared:.2f} vs actual {actual:.2f}/month -- gap of {gap_pct*100:.1f}%",
    }


if __name__ == "__main__":
    result = check_income_upi("P006_income_mismatch")
    print(result)
