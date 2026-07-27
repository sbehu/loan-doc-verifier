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


def save_upload(uploaded_file, dest_path: Path) -> None:
    """Re-encodes whatever format was uploaded (jpg, png, ...) as PNG at
    the exact filename the checks package expects."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(uploaded_file).convert("RGB")
    img.save(dest_path, "PNG")


def run_verification(uploaded: dict) -> tuple[str, dict]:
    persona_id = f"_uploads/{uuid.uuid4().hex[:10]}"
    doc_dir = base_dir / "data" / "documents" / persona_id
    for key, value in uploaded.items():
        if key == "upi_statement":
            for i, f in enumerate(value, 1):
                save_upload(f, doc_dir / f"upi_statement_{i}.png")
        else:
            save_upload(value, doc_dir / f"{key}.png")

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
cols = st.columns(2)
for i, (key, label) in enumerate(COMMON_DOCS):
    with cols[i % 2]:
        uploaded[key] = st.file_uploader(label, type=["png", "jpg", "jpeg"], key=key)

with cols[len(COMMON_DOCS) % 2]:
    if employment_type == "Salaried":
        uploaded["salary_slip"] = st.file_uploader(
            "Salary Slip", type=["png", "jpg", "jpeg"], key="salary_slip"
        )
    else:
        uploaded["upi_statement"] = st.file_uploader(
            "UPI Statement(s) -- upload one per account if the applicant has multiple",
            type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="upi_statement",
        )

required_keys = [k for k, _ in COMMON_DOCS] + (
    ["salary_slip"] if employment_type == "Salaried" else ["upi_statement"]
)
all_present = all(uploaded.get(k) for k in required_keys)

if st.button("Run Verification", disabled=not all_present, type="primary"):
    persona_id, outcome = run_verification(uploaded)
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
