"""
Identity consistency check.
Verifies the applicant's name, PAN, and Aadhaar number declared on the
application form actually match what's printed on the Aadhaar card, PAN
card, and electricity bill. Nothing else in this pipeline ties a
document to a specific identity by name/ID number -- photo_match checks
the FACE, signature checks handwriting, but a borrowed or fabricated ID
document with a slightly different name/number on it would otherwise
sail through untouched.

Comparison is deterministic string matching after normalization (case,
whitespace, honorifics) -- not an LLM fuzzy judgment call. Legal
identity numbers and names need an exact match, not "plausibly
similar": a single-letter difference in a name on a PAN card is exactly
the kind of real identity-mismatch signal that must never be waved
through as OCR noise (this is what the P005_identity_mismatch persona
actually tests -- PAN card reads "Gavin Genesan" vs. the applicant's
real name "Gavin Ganesan").

Simplifying assumption: the electricity bill is expected to be in the
applicant's own name. Real households sometimes have utility bills in a
family member's name -- out of scope for this check.
"""

import base64
import json
import re
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

base_dir = Path(__file__).resolve().parent.parent
load_dotenv(base_dir / ".env")
client = OpenAI()

HONORIFICS = {"MR", "MRS", "MS", "DR", "SHRI", "SMT", "KUM"}

FORM_PROMPT = (
    "This is a loan application form. Extract the 'Applicant Name', 'PAN', "
    "and 'Aadhaar' fields exactly as printed. Return ONLY JSON: "
    '{"name": ..., "pan": ..., "aadhaar": ...}'
)
AADHAAR_PROMPT = (
    "This is an Aadhaar card. Extract the full name and the 12-digit "
    "Aadhaar number exactly as printed. Return ONLY JSON: "
    '{"name": ..., "aadhaar_number": ...}'
)
PAN_PROMPT = (
    "This is a PAN card. Extract the full name and the PAN number exactly "
    'as printed. Return ONLY JSON: {"name": ..., "pan_number": ...}'
)
BILL_PROMPT = (
    "This is an electricity bill. Extract the consumer/account holder "
    "name printed on it -- not the address. Return ONLY JSON: "
    '{"name": ...}'
)


def _encode(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _extract(image_path: Path, prompt: str, retries: int = 2) -> dict:
    """Occasionally gpt-4o returns an empty message.content for a vision +
    JSON-mode call (observed directly on this project's own Aadhaar-card
    extraction -- same image, same prompt, succeeded once and returned
    empty the next time). Not a policy refusal (confirmed: the same call
    succeeds on retry), just API-level flakiness -- so retry a couple of
    times before giving up, rather than silently treating a missing field
    as "no mismatch found"."""
    img = Image.open(image_path).convert("RGB")
    last_error = None
    for attempt in range(retries + 1):
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode(img)}"}},
            ]}],
        )
        content = resp.choices[0].message.content
        if content:
            return json.loads(content)
        last_error = "empty response content"
    raise RuntimeError(
        f"gpt-4o returned empty content for {image_path.name} after "
        f"{retries + 1} attempts ({last_error})"
    )


def _normalize_name(name: str) -> str:
    if not name:
        return ""
    tokens = re.sub(r"[^A-Za-z\s]", " ", name).upper().split()
    tokens = [t for t in tokens if t not in HONORIFICS]
    return " ".join(tokens)


def _normalize_id(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", value).upper()


# Vision-model OCR occasionally misreads a single visually-similar
# character (confirmed directly: P006's real PAN is "DGUFI6975Z", but one
# extraction of the printed card came back "DGUF16975Z" -- position 5
# read as digit '1' instead of letter 'I'). PAN numbers have a fixed,
# public format (5 letters, 4 digits, 1 letter -- always in that order),
# so rather than loosening the match, we use that known structure to
# correct misreads deterministically: a position that must be a letter
# but came back as a digit (or vice versa) gets swapped for its
# documented look-alike. This can only fix a reading, never hide a real
# mismatch, since it only touches characters that violate PAN's format
# rule in the first place.
_DIGIT_TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"}
_LETTER_TO_DIGIT = {"O": "0", "I": "1", "S": "5", "B": "8", "Z": "2"}


def _correct_pan(pan: str) -> str:
    if len(pan) != 10:
        return pan  # doesn't match PAN's length -- can't safely position-correct
    chars = list(pan)
    for i, c in enumerate(chars):
        expect_letter = i < 5 or i == 9
        if expect_letter and c in _DIGIT_TO_LETTER:
            chars[i] = _DIGIT_TO_LETTER[c]
        elif not expect_letter and c in _LETTER_TO_DIGIT:
            chars[i] = _LETTER_TO_DIGIT[c]
    return "".join(chars)


def _correct_aadhaar(aadhaar: str) -> str:
    if len(aadhaar) != 12:
        return aadhaar  # doesn't match Aadhaar's length -- can't safely correct
    return "".join(_LETTER_TO_DIGIT.get(c, c) for c in aadhaar)


def check_identity(persona_id: str) -> dict:
    doc_dir = base_dir / "data" / "documents" / persona_id
    form_path = doc_dir / "application_form.png"
    aadhaar_path = doc_dir / "aadhaar.png"
    pan_path = doc_dir / "pan.png"
    bill_path = doc_dir / "electricity_bill.png"

    if not form_path.exists():
        return {
            "persona_id": persona_id,
            "check": "identity_consistency",
            "flagged": False,
            "detail": "no application form present -- check not applicable",
        }

    form = _extract(form_path, FORM_PROMPT)
    form_name = _normalize_name(form.get("name"))
    form_pan = _correct_pan(_normalize_id(form.get("pan")))
    form_aadhaar = _correct_aadhaar(_normalize_id(form.get("aadhaar")))

    mismatches = []

    if aadhaar_path.exists():
        aadhaar_doc = _extract(aadhaar_path, AADHAAR_PROMPT)
        aadhaar_name = _normalize_name(aadhaar_doc.get("name"))
        aadhaar_number = _correct_aadhaar(_normalize_id(aadhaar_doc.get("aadhaar_number")))
        if aadhaar_name and form_name and aadhaar_name != form_name:
            mismatches.append(
                f"name on Aadhaar card ('{aadhaar_doc.get('name')}') does not match "
                f"application form ('{form.get('name')}')"
            )
        if aadhaar_number and form_aadhaar and aadhaar_number != form_aadhaar:
            mismatches.append(
                f"Aadhaar number on card ({aadhaar_doc.get('aadhaar_number')}) does not "
                f"match application form ({form.get('aadhaar')})"
            )

    if pan_path.exists():
        pan_doc = _extract(pan_path, PAN_PROMPT)
        pan_name = _normalize_name(pan_doc.get("name"))
        pan_number = _correct_pan(_normalize_id(pan_doc.get("pan_number")))
        if pan_name and form_name and pan_name != form_name:
            mismatches.append(
                f"name on PAN card ('{pan_doc.get('name')}') does not match "
                f"application form ('{form.get('name')}')"
            )
        if pan_number and form_pan and pan_number != form_pan:
            mismatches.append(
                f"PAN number on card ({pan_doc.get('pan_number')}) does not match "
                f"application form ({form.get('pan')})"
            )

    if bill_path.exists():
        bill_doc = _extract(bill_path, BILL_PROMPT)
        bill_name = _normalize_name(bill_doc.get("name"))
        if bill_name and form_name and bill_name != form_name:
            mismatches.append(
                f"account holder name on electricity bill ('{bill_doc.get('name')}') "
                f"does not match application form ('{form.get('name')}')"
            )

    return {
        "persona_id": persona_id,
        "check": "identity_consistency",
        "flagged": len(mismatches) > 0,
        "detail": "; ".join(mismatches) if mismatches else "name, PAN, and Aadhaar consistent across all documents",
    }


if __name__ == "__main__":
    result = check_identity("P005_identity_mismatch")
    print(result)
