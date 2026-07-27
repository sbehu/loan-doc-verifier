"""
Document recycling check.
Two-stage: (1) ChromaDB + CLIP embeddings cheaply find each persona's
single closest signature match across the whole applicant pool -- this
is the part that scales to millions of records. (2) Only that top
candidate pair gets a precise vision-LLM comparison (same mechanism as
the signature-forgery check) to confirm whether it's actually a real
match.

Unlike every other check, this one can't answer a question about a
single persona in isolation -- it has to look at the whole applicant
pool at once to find matches BETWEEN personas. So instead of a
per-persona function, this exposes find_recycled_signatures(personas),
meant to be run ONCE, up front, before processing any individual
persona. It returns a dict keyed by persona_id, with a result for
every persona in the same {persona_id, check, flagged, detail} shape
as the other checks -- so main.py can still do a simple per-persona
lookup after this one batch pass runs.
"""

import base64
import json
from io import BytesIO
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from PIL import Image

base_dir = Path(__file__).resolve().parent.parent
load_dotenv(base_dir / ".env")

embed_model = SentenceTransformer("clip-ViT-B-32")
llm_client = OpenAI()

CONFIRM_PROMPT = (
    "You are comparing two signature images: SIGNATURE_A and SIGNATURE_B. "
    "Judge whether they show the same handwriting style, allowing for natural "
    "variation in pen pressure, slight wobble, or minor size differences, but "
    "flagging real structural differences (different letter shapes, different "
    "stroke connections, different slant). Return ONLY JSON: "
    '{"same_person": true/false, "confidence": 0-100, "reasoning": "one sentence"}'
)


def embed_image(path: Path):
    return embed_model.encode(Image.open(path).convert("RGB")).tolist()


def _encode(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def confirm_match(persona_a: str, persona_b: str) -> dict:
    doc_dir = base_dir / "data" / "documents"
    img_a = Image.open(doc_dir / persona_a / "signature_reference.png").convert("RGB")
    img_b = Image.open(doc_dir / persona_b / "signature_reference.png").convert("RGB")

    resp = llm_client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": CONFIRM_PROMPT},
            {"type": "text", "text": "SIGNATURE_A:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode(img_a)}"}},
            {"type": "text", "text": "SIGNATURE_B:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode(img_b)}"}},
        ]}],
    )
    return json.loads(resp.choices[0].message.content)


def find_recycled_signatures(personas: list) -> dict:
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="signatures", metadata={"hnsw:space": "cosine"})

    # Stage 1: embed every persona's signature into the vector store
    for persona in personas:
        sig_path = base_dir / "data" / "documents" / persona["persona_id"] / "signature_reference.png"
        collection.add(
            ids=[persona["persona_id"]],
            embeddings=[embed_image(sig_path)],
            metadatas=[{"persona_id": persona["persona_id"]}],
        )

    # Stage 1 continued: for each persona, find its single closest OTHER match
    candidate_pairs = set()
    for persona in personas:
        sig_path = base_dir / "data" / "documents" / persona["persona_id"] / "signature_reference.png"
        results = collection.query(query_embeddings=[embed_image(sig_path)], n_results=2)
        for other_id in results["ids"][0]:
            if other_id != persona["persona_id"]:
                candidate_pairs.add(tuple(sorted([persona["persona_id"], other_id])))
                break

    # Stage 2: precise LLM confirmation, only on the shortlisted candidates
    flagged_pairs = {}  # persona_id -> (matched_with, reasoning, confidence)
    for persona_a, persona_b in candidate_pairs:
        try:
            result = confirm_match(persona_a, persona_b)
        except TypeError:
            continue  # model returned no content, skip this pair
        if result["same_person"]:
            flagged_pairs[persona_a] = (persona_b, result["reasoning"], result["confidence"])
            flagged_pairs[persona_b] = (persona_a, result["reasoning"], result["confidence"])

    results_by_persona = {}
    for persona in personas:
        pid = persona["persona_id"]
        if pid in flagged_pairs:
            matched_with, reasoning, confidence = flagged_pairs[pid]
            results_by_persona[pid] = {
                "persona_id": pid,
                "check": "document_recycling",
                "flagged": True,
                "detail": f"signature matches {matched_with}: {reasoning} (confidence: {confidence}%)",
            }
        else:
            results_by_persona[pid] = {
                "persona_id": pid,
                "check": "document_recycling",
                "flagged": False,
                "detail": "signature is unique across the applicant pool",
            }
    return results_by_persona


def build_pool_index(personas: list):
    """Embeds the known applicant pool's signatures into a fresh Chroma
    collection and returns it. Meant to be built ONCE (e.g. cached by the
    caller) and reused for every new upload checked via
    check_recycling_against_pool, rather than re-embedding the whole known
    pool on every single new-persona check."""
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="signatures_pool", metadata={"hnsw:space": "cosine"}
    )
    for persona in personas:
        sig_path = base_dir / "data" / "documents" / persona["persona_id"] / "signature_reference.png"
        collection.add(
            ids=[persona["persona_id"]],
            embeddings=[embed_image(sig_path)],
            metadatas=[{"persona_id": persona["persona_id"]}],
        )
    return collection


def check_recycling_against_pool(new_persona_id: str, pool_collection) -> dict:
    """Same two-stage idea as find_recycled_signatures, but for a single
    brand-new persona (e.g. a fresh upload) that isn't part of the known
    pool used to build pool_collection. Finds the closest known match,
    then confirms with the vision LLM."""
    sig_path = base_dir / "data" / "documents" / new_persona_id / "signature_reference.png"
    results = pool_collection.query(query_embeddings=[embed_image(sig_path)], n_results=1)

    if not results["ids"][0]:
        return {
            "persona_id": new_persona_id,
            "check": "document_recycling",
            "flagged": False,
            "detail": "no known signatures to compare against",
        }

    top_match_id = results["ids"][0][0]
    try:
        result = confirm_match(new_persona_id, top_match_id)
    except TypeError:
        return {
            "persona_id": new_persona_id,
            "check": "document_recycling",
            "flagged": False,
            "detail": "confirmation step returned no result -- treating as unique",
        }

    if result["same_person"]:
        return {
            "persona_id": new_persona_id,
            "check": "document_recycling",
            "flagged": True,
            "detail": f"signature matches {top_match_id}: {result['reasoning']} (confidence: {result['confidence']}%)",
        }
    return {
        "persona_id": new_persona_id,
        "check": "document_recycling",
        "flagged": False,
        "detail": "signature is unique against the known applicant pool",
    }


if __name__ == "__main__":
    personas = json.loads((base_dir / "data" / "ground_truth" / "personas.json").read_text())
    results = find_recycled_signatures(personas)
    for pid, result in results.items():
        marker = "FLAGGED" if result["flagged"] else "ok"
        print(f"{pid:35s} [{marker}] {result['detail']}")
