"""
Photo match check (Aadhaar ID photo vs. live selfie).
Uses AWS Rekognition's CompareFaces API -- a real, ready-made face
matching service rather than asking a vision LLM to make a judgment
call. Rekognition auto-detects the face within each image (no manual
cropping needed) and returns a similarity score between the largest
face in the source image and faces found in the target image.
"""

from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

base_dir = Path(__file__).resolve().parent.parent
load_dotenv(base_dir / ".env")

rekognition = boto3.client("rekognition")

SIMILARITY_THRESHOLD = 80


def check_photo_match(persona_id: str) -> dict:
    doc_dir = base_dir / "data" / "documents" / persona_id
    aadhaar_path = doc_dir / "aadhaar.png"
    selfie_path = doc_dir / "selfie.png"

    if not aadhaar_path.exists() or not selfie_path.exists():
        return {
            "persona_id": persona_id,
            "check": "photo_match",
            "flagged": False,
            "detail": "aadhaar or selfie image missing -- check not applicable",
        }

    with open(aadhaar_path, "rb") as f:
        source_bytes = f.read()
    with open(selfie_path, "rb") as f:
        target_bytes = f.read()

    try:
        resp = rekognition.compare_faces(
            SourceImage={"Bytes": source_bytes},
            TargetImage={"Bytes": target_bytes},
            SimilarityThreshold=SIMILARITY_THRESHOLD,
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "InvalidParameterException":
            return {
                "persona_id": persona_id,
                "check": "photo_match",
                "flagged": True,
                "detail": "no face detected in the ID photo or the selfie -- cannot verify, flagging for manual review",
            }
        raise

    matches = resp.get("FaceMatches", [])
    if not matches:
        return {
            "persona_id": persona_id,
            "check": "photo_match",
            "flagged": True,
            "detail": f"no face match found above {SIMILARITY_THRESHOLD}% similarity between ID photo and selfie",
        }

    similarity = matches[0]["Similarity"]
    return {
        "persona_id": persona_id,
        "check": "photo_match",
        "flagged": False,
        "detail": f"ID photo and selfie match at {similarity:.1f}% similarity",
    }


if __name__ == "__main__":
    result = check_photo_match("P008_photo_mismatch")
    print(result)
