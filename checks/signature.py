"""
Signature forgery check.
Compares the signature on application_form.png against
signature_reference.png using a vision model to judge handwriting
similarity, allowing for natural variation rather than requiring a
pixel-exact match.

Returns the raw model judgment (same_person, confidence, reasoning)
rather than a single flagged bool -- the orchestrator decides severity
based on confidence: low confidence in a mismatch is treated as clean
(natural signature variation), medium confidence triggers further
investigation/resubmission, high confidence is a clear reject.
"""

import base64
import json
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

base_dir = Path(__file__).resolve().parent.parent
load_dotenv(base_dir / ".env")
client = OpenAI()

SIGNATURE_CROP_BOX = (30, 424, 330, 524)

PROMPT = (
    "You are comparing two signature images: REFERENCE (a known genuine signature "
    "on file) and SUBMITTED (a new signature to verify). Judge whether they were "
    "written by the same person, allowing for natural variation in handwriting "
    "(pen pressure, slight wobble, minor size differences) but flagging real "
    "structural differences (different letter shapes, different stroke connections, "
    "different slant). Return ONLY JSON: "
    '{"same_person": true/false, "confidence": 0-100, "reasoning": "one sentence"}'
)


def _encode(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def check_signature(persona_id: str) -> dict:
    doc_dir = base_dir / "data" / "documents" / persona_id
    reference = Image.open(doc_dir / "signature_reference.png").convert("RGB")
    form = Image.open(doc_dir / "application_form.png").convert("RGB")
    signature_on_form = form.crop(SIGNATURE_CROP_BOX)

    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "text", "text": "REFERENCE:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode(reference)}"}},
            {"type": "text", "text": "SUBMITTED:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode(signature_on_form)}"}},
        ]}],
    )
    result = json.loads(resp.choices[0].message.content)

    return {
        "persona_id": persona_id,
        "check": "signature_forgery",
        "same_person": result["same_person"],
        "confidence": result["confidence"],
        "reasoning": result["reasoning"],
    }


if __name__ == "__main__":
    result = check_signature("P007_signature_forged")
    print(result)
