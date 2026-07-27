"""
Address consistency check (application form vs. address proof).
Sends the full application form and full electricity bill to the vision
model, which locates the address on each and judges whether they match.
No assumptions about document layout -- works from the documents alone.

Also checks the electricity bill's own recency: a bill that matches the
address text perfectly but was issued too long before the application
date isn't valid proof of CURRENT residence -- real underwriting
typically requires address proof issued within the last ~90 days. This
is a genuinely separate signal from text-matching and wasn't checked at
all before (the "stale address" persona surfaced the gap: same address
text on both documents, but the bill was issued 16+ months earlier).

Returns outcome: "clean" | "needs_resubmission" instead of a flagged
bool -- an address problem should route to "please resend your address
proof," not an automatic fraud rejection.
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

STALE_THRESHOLD_DAYS = 90

PROMPT = (
    "The first image is a loan APPLICATION FORM; the second is an ADDRESS PROOF "
    "document (electricity bill). Find the address on each, then judge whether "
    "they refer to the same physical address, allowing for minor formatting "
    "differences (abbreviations, punctuation, spacing) but flagging real "
    "differences (different house number, street, or city). Also find the "
    "application form's 'Application date' and the electricity bill's 'Bill "
    "Issue Date', and return each as YYYY-MM-DD (use day 01 if only a month "
    "and year are shown, e.g. 'June 2026' -> '2026-06-01'). Return ONLY JSON: "
    '{"same_address": true/false, "confidence": 0-100, "reasoning": "one sentence", '
    '"application_date": "YYYY-MM-DD", "bill_issue_date": "YYYY-MM-DD"}'
)


def _encode(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def check_address(persona_id: str) -> dict:
    doc_dir = base_dir / "data" / "documents" / persona_id
    form = Image.open(doc_dir / "application_form.png").convert("RGB")
    bill = Image.open(doc_dir / "electricity_bill.png").convert("RGB")

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode(form)}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode(bill)}"}},
        ]}],
    )
    result = json.loads(resp.choices[0].message.content)

    reasons = []
    outcome = "clean"

    if not result["same_address"]:
        outcome = "needs_resubmission"
        reasons.append(f"{result['reasoning']} (confidence: {result['confidence']}%)")

    stale_days = None
    try:
        app_date = date.fromisoformat(result["application_date"])
        bill_date = date.fromisoformat(result["bill_issue_date"])
        stale_days = (app_date - bill_date).days
    except (KeyError, TypeError, ValueError):
        pass  # couldn't parse one of the dates -- skip staleness check, don't guess

    if stale_days is not None and stale_days > STALE_THRESHOLD_DAYS:
        outcome = "needs_resubmission"
        reasons.append(
            f"electricity bill issued {result['bill_issue_date']} is {stale_days} "
            f"days old relative to the application date ({result['application_date']}) "
            f"-- too old to serve as proof of current address"
        )

    if not reasons:
        reasons.append(f"{result['reasoning']} (confidence: {result['confidence']}%)")

    return {
        "persona_id": persona_id,
        "check": "address_consistency",
        "outcome": outcome,
        "detail": "; ".join(reasons),
    }


if __name__ == "__main__":
    result = check_address("P001_clean_salaried")
    print(result)
