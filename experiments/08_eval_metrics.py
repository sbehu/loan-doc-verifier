"""
Evaluation metrics for the verification pipeline.

There's no held-out "real world" ground truth for a fraud-detection
system like this -- so the ground truth used here is the one built
into the synthetic dataset itself: data/ground_truth/personas.json
tags each persona with a `fraud_type` (None for the 5 clean personas,
a specific string like "signature_forged" for the 9 fraud personas).
That's exactly what a held-out labeled eval set looks like in a real
fraud team, just synthetic instead of historical.

Two things get measured:

1. Overall confusion matrix -- "should this application have been
   flagged at all" (fraud_type is not None) vs. "was it" (final
   verdict != APPROVE). This is the number an interviewer or a risk
   stakeholder actually cares about: false-positive rate (clean
   applicants wrongly flagged -- a customer-friction/ops cost) and
   false-negative rate (fraud that slipped through -- a credit-risk
   cost). These trade off against each other; which balance is
   acceptable is a policy call, not an engineering one, but you can't
   make that policy call without these two numbers.

2. Per-persona, per-check attribution -- for each fraud persona, did
   the ONE check specifically designed to catch that fraud type
   actually fire? This is the more actionable table for debugging:
   the overall confusion matrix tells you something's wrong, this
   tells you which check.

What this deliberately does NOT do: score the two LLM-generated
free-text reasoning steps (the "dig deeper" investigation step, and
app.py's chat Q&A). Those aren't classification outputs, so
precision/recall doesn't apply -- they'd need a groundedness-style
check (does the generated reasoning actually follow from the
underlying data) instead, most likely via human-labeled spot checks
or an LLM-as-judge grader. That's a separate, not-yet-built piece of
this eval story -- noted here rather than silently left out.
"""

import json
import sys
from pathlib import Path

from tqdm import tqdm

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))  # main.py lives at the project root, but
                                    # a script run as experiments/08_....py
                                    # only gets its own directory on
                                    # sys.path by default, not the root.

from main import personas, process_persona, get_recycling_results

# Which single check this project's own persona design intends to be
# the one that catches each fraud type. Used only for the per-check
# attribution table below -- an approximation of intent, not a claim
# that it's the ONLY check that could ever catch that fraud type in
# principle (e.g. a forged signature could in theory also show up as
# an identity mismatch).
FRAUD_TYPE_TO_CHECK = {
    "doc_tamper_careless": "balance",
    "doc_tamper_careful": "balance",
    "identity_mismatch": "identity",
    "income_mismatch": "income_salary",
    "signature_forged": "signature",
    "photo_mismatch": "photo_match",
    "fully_fabricated": None,  # not targeted at one specific check -- a
                               # broadly fabricated identity that's meant
                               # to be caught by the combination of checks,
                               # not any single one
    "stale_address": "address",
    "document_recycling": "recycling",
}


def _check_fired(check_result: dict) -> bool:
    """A check 'firing' means it raised something -- either a hard flag
    or (for identity) a needs_review near-miss. Being lenient here
    (counting needs_review as a hit) matches how build_verdict treats
    it: identity's near-miss tier still surfaces the case for review
    rather than silently passing it."""
    if check_result is None:
        return False
    if check_result.get("flagged"):
        return True
    if check_result.get("needs_review"):
        return True
    return False


def main():
    print("precomputing recycling detection across the applicant pool...")
    get_recycling_results()

    rows = []
    for persona in tqdm(personas, desc="personas", unit="persona"):
        persona_id = persona["persona_id"]
        outcome = process_persona(persona_id)
        verdict = outcome["verdict"]["verdict"]
        results = outcome["results"]

        truth_positive = persona["fraud_type"] is not None
        pred_positive = verdict != "APPROVE"

        if truth_positive and pred_positive:
            category = "TP"
        elif truth_positive and not pred_positive:
            category = "FN"
        elif not truth_positive and pred_positive:
            category = "FP"
        else:
            category = "TN"

        expected_check = FRAUD_TYPE_TO_CHECK.get(persona["fraud_type"]) if truth_positive else None
        check_hit = _check_fired(results.get(expected_check)) if expected_check else None

        rows.append({
            "persona_id": persona_id,
            "scenario": persona["scenario"],
            "fraud_type": persona["fraud_type"],
            "truth_positive": truth_positive,
            "verdict": verdict,
            "pred_positive": pred_positive,
            "category": category,
            "expected_check": expected_check,
            "check_hit": check_hit,
        })

    tp = sum(1 for r in rows if r["category"] == "TP")
    fp = sum(1 for r in rows if r["category"] == "FP")
    fn = sum(1 for r in rows if r["category"] == "FN")
    tn = sum(1 for r in rows if r["category"] == "TN")

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")

    print("\n" + "=" * 100)
    print(f"{'persona_id':32s} {'fraud_type':22s} {'verdict':22s} {'category':10s} {'expected_check':16s} {'check_hit':10s}")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['persona_id']:32s} {str(r['fraud_type']):22s} {r['verdict']:22s} "
            f"{r['category']:10s} {str(r['expected_check']):16s} {str(r['check_hit']):10s}"
        )
    print("=" * 100)

    print("\nConfusion matrix (ground truth = fraud_type is not None, prediction = verdict != APPROVE):")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  precision = {precision:.2f}   (of applications flagged, fraction that were actually fraud)")
    print(f"  recall    = {recall:.2f}   (of actual fraud, fraction that got flagged)")
    print(f"  f1        = {f1:.2f}")
    print(f"  FPR       = {fpr:.2f}   (of clean applicants, fraction wrongly flagged -- customer-friction cost)")
    print(f"  FNR       = {fnr:.2f}   (of actual fraud, fraction that slipped through -- credit-risk cost)")

    misses = [r for r in rows if r["category"] == "FN"]
    if misses:
        print("\nFalse negatives (known gaps, not silently swept under the rug):")
        for r in misses:
            print(f"  - {r['persona_id']} ({r['fraud_type']}) -> verdict was {r['verdict']}")

    out_path = base_dir / "experiments" / "eval_results.json"
    out_path.write_text(json.dumps({
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "metrics": {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr, "fnr": fnr},
        "rows": rows,
    }, indent=2))
    print(f"\nfull results written to {out_path}")


if __name__ == "__main__":
    main()
