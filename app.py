"""
Streamlit UI for loan_doc_verifier.

Lets a loan officer upload a brand-new applicant's documents, run the
full verification pipeline (main.process_persona) unchanged, see the
verdict and underlying check results, and ask follow-up questions about
the result in a chat interface grounded in the actual pipeline output.

Uploaded documents are saved into data/documents/_uploads/<random_id>/
using the same filenames the checks package already expects
(aadhaar.png, application_form.png, etc.) -- so process_persona() needs
no changes at all to handle a brand-new applicant vs. one of the 14
pre-built test personas.
"""

import json
import uuid
from pathlib import Path

import fitz  # PyMuPDF
import streamlit as st
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

from main import process_persona, get_recycling_results

base_dir = Path(__file__).resolve().parent
load_dotenv(base_dir / ".env")

st.set_page_config(page_title="Loan Document Verifier", layout="wide")
st.title("Loan Document Verifier")

chat_client = OpenAI()

COMMON_DOCS = [
    ("aadhaar", "Aadhaar Card"),
    ("application_form", "Application Form"),
    ("bank_statement", "Bank Statement"),
    ("electricity_bill", "Electricity Bill (address proof)"),
    ("pan", "PAN Card"),
    ("selfie", "Selfie"),
    ("signature_reference", "Signature Reference (bank specimen)"),
]

VERDICT_COLOR = {
    "APPROVE": "green",
    "REJECT": "red",
    "WARNING_RESUBMISSION": "orange",
    "CONDITIONAL_APPROVAL": "orange",
}


def _pdf_to_images(pdf_bytes: bytes, dpi: int = 200, password: str | None = None) -> list[Image.Image]:
    """Renders every page of a PDF to a PIL Image. Real-world bank
    statements, salary slips, and UPI exports are very commonly PDFs --
    none of the checks know how to read a PDF directly (they all open
    the document as a single image), so any PDF upload is converted
    here before it ever reaches the pipeline.

    Bank statement PDFs are frequently password-protected -- the
    password is typed directly into this app (never sent to Claude),
    and used only locally to unlock the PDF for rendering."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.needs_pass:
        if not password or not doc.authenticate(password):
            raise ValueError("PDF is password-protected and the password provided didn't unlock it")
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return images


def _stack_vertically(images: list[Image.Image]) -> Image.Image:
    width = max(img.width for img in images)
    total_height = sum(img.height for img in images)
    canvas = Image.new("RGB", (width, total_height), "white")
    y = 0
    for img in images:
        canvas.paste(img, (0, y))
        y += img.height
    return canvas


MAX_SINGLE_PAGE_DIMENSION = 2000


def save_upload(uploaded_file, dest_path: Path, pdf_password: str | None = None) -> None:
    """Re-encodes whatever format was uploaded (jpg, png, pdf) as PNG at
    the exact filename the checks package expects. PDFs get converted
    to image(s) first -- bank_statement and upi_statement pages are
    stacked into one tall image (those checks already chunk tall images
    by height, so this fits the existing design), everything else keeps
    only the first page, since those checks treat the document as a
    single image.

    Non-statement documents (selfie, aadhaar, PAN, etc.) get capped to
    MAX_SINGLE_PAGE_DIMENSION on their longest side before saving.
    Real phone-camera photos are often several MP -- re-encoded as
    lossless PNG that can exceed AWS Rekognition's hard 5MB-per-image
    limit (hit directly: a real selfie upload triggered
    "targetImage.bytes" > 5242880 bytes on compare_faces). Bank/UPI
    statements are deliberately excluded from this cap -- their own
    extraction pipeline already manages resolution for OCR/table
    detection, and shrinking them here could hurt text readability."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    is_pdf = uploaded_file.name.lower().endswith(".pdf") or uploaded_file.type == "application/pdf"
    stackable = dest_path.stem.startswith("bank_statement") or dest_path.stem.startswith("upi_statement")

    if is_pdf:
        pages = _pdf_to_images(uploaded_file.getvalue(), password=pdf_password)
        img = _stack_vertically(pages) if stackable else pages[0]
    else:
        img = Image.open(uploaded_file).convert("RGB")

    img = img.convert("RGB")
    if not stackable and max(img.size) > MAX_SINGLE_PAGE_DIMENSION:
        img.thumbnail((MAX_SINGLE_PAGE_DIMENSION, MAX_SINGLE_PAGE_DIMENSION), Image.LANCZOS)

    img.save(dest_path, "PNG")


def run_verification(uploaded: dict, pdf_passwords: dict) -> tuple[str, dict]:
    persona_id = f"_uploads/{uuid.uuid4().hex[:10]}"
    doc_dir = base_dir / "data" / "documents" / persona_id
    for key, value in uploaded.items():
        if key == "upi_statement":
            for i, f in enumerate(value, 1):
                save_upload(f, doc_dir / f"upi_statement_{i}.png")
        else:
            save_upload(value, doc_dir / f"{key}.png", pdf_password=pdf_passwords.get(key))

    progress = st.progress(0, text="Saving uploaded documents...")

    # Recycling detection loads an embedding model and compares against
    # the whole known applicant pool -- only genuinely slow on the very
    # first run of the app (cached after that), but with no feedback
    # here that first run looks identical to the page being frozen.
    progress.progress(15, text="Preparing recycling-detection index (one-time cost on first run)...")
    get_recycling_results()

    progress.progress(40, text="Running document checks (signature, balance, address, photo match, income)...")
    outcome = process_persona(persona_id)

    progress.progress(100, text="Done.")
    return persona_id, outcome


# ---- Step 1: employment type ----
employment_type = st.radio("Employment type", ["Salaried", "Self-employed"], horizontal=True)

st.subheader("Upload documents")
uploaded = {}
pdf_passwords = {}
cols = st.columns(2)
for i, (key, label) in enumerate(COMMON_DOCS):
    with cols[i % 2]:
        uploaded[key] = st.file_uploader(label, type=["png", "jpg", "jpeg", "pdf"], key=key)
        if key == "bank_statement":
            pdf_passwords[key] = st.text_input(
                "Bank statement PDF password (leave blank if not password-protected)",
                type="password", key="bank_statement_pw",
            )

with cols[len(COMMON_DOCS) % 2]:
    if employment_type == "Salaried":
        uploaded["salary_slip"] = st.file_uploader(
            "Salary Slip", type=["png", "jpg", "jpeg", "pdf"], key="salary_slip"
        )
    else:
        uploaded["upi_statement"] = st.file_uploader(
            "UPI Statement(s) -- upload one per account if the applicant has multiple",
            type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True, key="upi_statement",
        )

required_keys = [k for k, _ in COMMON_DOCS] + (
    ["salary_slip"] if employment_type == "Salaried" else ["upi_statement"]
)
all_present = all(uploaded.get(k) for k in required_keys)

if st.button("Run Verification", disabled=not all_present, type="primary"):
    persona_id, outcome = run_verification(uploaded, pdf_passwords)
    st.session_state["persona_id"] = persona_id
    st.session_state["outcome"] = outcome
    st.session_state["messages"] = []

if not all_present:
    st.caption("Upload all required documents to enable verification.")

# ---- Results ----
if "outcome" in st.session_state:
    outcome = st.session_state["outcome"]
    verdict = outcome["verdict"]["verdict"]
    reasons = outcome["verdict"]["reasons"]
    color = VERDICT_COLOR.get(verdict, "gray")

    st.divider()
    st.markdown(f"## Verdict: :{color}[{verdict}]")
    for r in reasons:
        st.write(f"- {r}")

    with st.expander("Full check results"):
        st.json(outcome["results"])
    if outcome["investigations"]:
        with st.expander("Investigation notes (agentic dig-deeper step)"):
            st.json(outcome["investigations"])

    st.divider()
    st.subheader("Ask about this verdict")

    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a follow-up question about this verdict...")
    if question:
        st.session_state["messages"].append({"role": "user", "content": question})

        context = json.dumps(outcome, indent=2, default=str)
        system_prompt = (
            "You are assisting a loan officer reviewing an automated document-"
            "verification verdict. Below is the full JSON output from the "
            "verification pipeline for this applicant: individual check "
            "results, any deeper investigation notes, and the final verdict. "
            "Answer the loan officer's questions using ONLY this data -- do "
            "not invent facts that aren't present in it. Be concise and "
            "direct.\n\n" + context
        )
        messages = [{"role": "system", "content": system_prompt}] + st.session_state["messages"]
        resp = chat_client.chat.completions.create(model="gpt-4o", messages=messages)
        answer = resp.choices[0].message.content
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.rerun()
