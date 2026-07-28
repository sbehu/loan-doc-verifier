"""
main.py - orchestrator entry point.
Runs all per-persona checks for a single persona_id, collects raw
results, triggers the "dig deeper" investigation step for borderline
income cases, and applies a fixed verdict rule engine on top -- see
project notes for why the final verdict is plain code, not an LLM call
(auditability), while the investigation step is a genuine LLM
reasoning call (the one real agentic piece of this pipeline).

Recycling is handled differently from the other 6 checks: it needs to
compare ALL personas' signatures against each other, not just one, so
it runs ONCE up front for the whole pool (find_recycled_signatures),
and run_all_checks() just looks up this persona's slice of that
already-computed result.
"""

import json
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from checks.signature import check_signature
from checks.balance import check_balance
from checks.address import check_address
from checks.identity import check_identity
from checks.photo_match import check_photo_match
from checks.income_upi import check_income_upi, extract_income_rows
from checks.income_salary import check_income_salary, get_bank_salary_credit
from checks.recycling import find_recycled_signatures, build_pool_index, check_recycling_against_pool

investigate_client = OpenAI()

base_dir = Path(__file__).resolve().parent
personas = json.loads((base_dir / "data" / "ground_truth" / "personas.json").read_text())

# Recycling needs to compare ALL personas' signatures against each other,
# so it's computed once for the whole pool rather than per-persona -- but
# lazily, on first actual use, NOT automatically at import time. main.py
# gets imported (not run as a script) by the Streamlit app, and running
# this expensive step (loading the embedding model, embedding all 14
# signatures, LLM confirmation calls) at import time meant the UI sat on
# a blank page with zero feedback before anything could even render.
_recycling_results = None
_recycling_pool_index = None


def get_recycling_results() -> dict:
    global _recycling_results
    if _recycling_results is None:
        _recycling_results = find_recycled_signatures(personas)
    return _recycling_results


def _get_recycling_result(persona_id: str) -> dict:
    recycling_results = get_recycling_results()
    if persona_id in recycling_results:
        return recycling_results[persona_id]
    global _recycling_pool_index
    if _recycling_pool_index is None:
        _recycling_pool_index = build_pool_index(personas)
    return check_recycling_against_pool(persona_id, _recycling_pool_index)


def _run_named(check_name: str, fn, *args):
    """Runs one check with a live start/done print (with elapsed seconds)
    so if something hangs, the last printed line tells us exactly which
    check it was stuck in -- no need to Ctrl+C to find out."""
    print(f"    running {check_name}...", end=" ", flush=True)
    start = time.time()
    result = fn(*args)
    print(f"done ({time.time() - start:.1f}s)")
    return result


def run_all_checks(persona_id: str) -> dict:
    return {
        "signature": _run_named("signature", check_signature, persona_id),
        "balance": _run_named("balance", check_balance, persona_id),
        "address": _run_named("address", check_address, persona_id),
        "identity": _run_named("identity", check_identity, persona_id),
        "photo_match": _run_named("photo_match", check_photo_match, persona_id),
        "income_upi": _run_named("income_upi", check_income_upi, persona_id),
        "income_salary": _run_named("income_salary", check_income_salary, persona_id),
        "recycling": _run_named("recycling", _get_recycling_result, persona_id),
    }


def needs_investigation(results: dict) -> list:
    """Returns the names of checks that fall into the ambiguous zone and
    need the agent to dig into raw data before a verdict can be decided."""
    triggers = []

    sig = results["signature"]
    if not sig["same_person"] and 50 <= sig["confidence"] <= 85:
        triggers.append("signature")

    upi = results["income_upi"]
    if upi["applicable"] and 0.10 < upi["gap_pct"] <= 0.20:
        triggers.append("income_upi")

    sal = results["income_salary"]
    if sal["applicable"]:
        gaps = [g for g in sal["gaps"].values() if g is not None]
        if any(0.10 < g <= 0.20 for g in gaps) and not any(g > 0.20 for g in gaps):
            triggers.append("income_salary")

    return triggers


INVESTIGATE_INCOME_PROMPT = (
    "A loan applicant declared a monthly income of {declared:.2f}. Based on "
    "their UPI transaction history, their actual average monthly income is "
    "{actual:.2f} -- a gap of {gap_pct:.1f}%. This gap isn't large enough to "
    "be obvious fraud, but isn't small enough to ignore either. Here is the "
    "raw list of credit (received) transactions behind that number:\n\n"
    "{rows_text}\n\n"
    "Look at this data and reason about whether the gap looks like a "
    "plausible, explainable pattern (e.g. a few unusually large one-off "
    "transactions, natural month-to-month variation) or looks suspicious "
    "(e.g. suspiciously round numbers, an unnatural cluster of transactions, "
    "patterns inconsistent with organic income). Return ONLY JSON: "
    "{{\"assessment\": \"explainable\" or \"suspicious\", \"reasoning\": "
    "\"2-3 sentences\"}}"
)

INVESTIGATE_SALARY_PROMPT = (
    "A loan applicant declared a monthly income of {declared:.2f}, and their "
    "salary slip shows net pay of {net_salary:.2f}. Based on their bank "
    "statement, here are the individual salary-credit deposits found:\n\n"
    "{rows_text}\n\n"
    "The average of these deposits doesn't closely match the declared/slip "
    "figures -- not a huge gap, but not negligible either. Look at this "
    "pattern and reason about whether it looks like a plausible, explainable "
    "situation (e.g. a raise mid-period, one irregular deposit, a bonus) or "
    "looks suspicious (e.g. deposits that don't look like real recurring "
    "salary payments). Return ONLY JSON: {{\"assessment\": \"explainable\" "
    "or \"suspicious\", \"reasoning\": \"2-3 sentences\"}}"
)


def investigate_income_upi(persona_id: str, result: dict) -> dict:
    doc_dir = base_dir / "data" / "documents" / persona_id
    statement_paths = sorted(doc_dir.glob("upi_statement*.png"))

    all_rows = []
    for path in statement_paths:
        rows, _ = extract_income_rows(path)
        all_rows.extend(rows)

    credit_rows = [r for r in all_rows if float(r["amount"]) > 0]
    rows_text = "\n".join(
        f"{r.get('date')}: {r.get('description')} +{float(r['amount']):.2f}"
        for r in sorted(credit_rows, key=lambda r: float(r["amount"]), reverse=True)
    )

    prompt = INVESTIGATE_INCOME_PROMPT.format(
        declared=result["declared"], actual=result["actual"],
        gap_pct=result["gap_pct"] * 100, rows_text=rows_text,
    )

    resp = investigate_client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(resp.choices[0].message.content)


def investigate_income_salary(persona_id: str, result: dict) -> dict:
    _, salary_rows = get_bank_salary_credit(persona_id)
    rows_text = "\n".join(
        f"{r.get('date')}: {r.get('description')} {float(r['amount']):.2f}"
        for r in salary_rows
    )

    prompt = INVESTIGATE_SALARY_PROMPT.format(
        declared=result["declared"], net_salary=result["net_salary"],
        rows_text=rows_text,
    )

    resp = investigate_client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(resp.choices[0].message.content)


def run_investigations(persona_id: str, results: dict) -> dict:
    """Runs the investigation step for every check that needs_investigation
    flagged, and returns a dict of {check_name: investigation_result}."""
    investigations = {}
    for check_name in needs_investigation(results):
        if check_name == "income_upi":
            investigations["income_upi"] = investigate_income_upi(persona_id, results["income_upi"])
        elif check_name == "income_salary":
            investigations["income_salary"] = investigate_income_salary(persona_id, results["income_salary"])
        # signature's borderline zone doesn't get a separate investigation
        # call -- there's no extra raw data to dig into beyond the visual
        # judgment already made, so it resolves straight to a verdict below.
    return investigations


def build_verdict(persona_id: str, results: dict, investigations: dict) -> dict:
    """Fixed rule engine -- takes all check results plus any investigation
    outcomes and produces the final verdict. Deliberately NOT an LLM call:
    loan verdicts need to be auditable and explainable.

    Rules: document recycling alone -> always REJECT. Otherwise, collect
    every other real issue present. Two or more issues together -> REJECT.
    Exactly one issue -> its own severity decides (balance/photo/income
    -> REJECT even alone, signature -> WARNING_RESUBMISSION, address ->
    CONDITIONAL_APPROVAL). Zero issues -> APPROVE.
    """
    if results["recycling"]["flagged"]:
        return {
            "persona_id": persona_id,
            "verdict": "REJECT",
            "reasons": ["document recycling: " + results["recycling"]["detail"]],
        }

    issues = []  # (check_name, severity_if_alone, reason)

    if results["balance"]["flagged"]:
        issues.append(("balance", "reject", "balance reconciliation failed: " + results["balance"]["detail"]))

    if results["photo_match"]["flagged"]:
        issues.append(("photo_match", "reject", "photo mismatch: " + results["photo_match"]["detail"]))

    identity = results["identity"]
    if identity["flagged"]:
        issues.append(("identity", "reject", "identity mismatch: " + identity["detail"]))
    elif identity.get("needs_review"):
        issues.append(("identity", "warning", "identity needs manual review: " + identity["detail"]))

    sig = results["signature"]
    if not sig["same_person"] and sig["confidence"] >= 50:
        issues.append(("signature", "warning", f"signature mismatch (confidence {sig['confidence']}%): {sig['reasoning']}"))

    upi = results["income_upi"]
    if upi["applicable"]:
        if upi["gap_pct"] > 0.20:
            issues.append(("income_upi", "reject", "income mismatch (UPI): " + upi["detail"]))
        elif upi["gap_pct"] > 0.10:
            inv = investigations.get("income_upi")
            if inv and inv["assessment"] == "suspicious":
                issues.append(("income_upi", "reject", f"income mismatch (UPI, investigated as suspicious): {inv['reasoning']}"))

    sal = results["income_salary"]
    if sal["applicable"]:
        gaps = [g for g in sal["gaps"].values() if g is not None]
        if any(g > 0.20 for g in gaps):
            issues.append(("income_salary", "reject", "income mismatch (salary): " + sal["detail"]))
        elif any(0.10 < g <= 0.20 for g in gaps):
            inv = investigations.get("income_salary")
            if inv and inv["assessment"] == "suspicious":
                issues.append(("income_salary", "reject", f"income mismatch (salary, investigated as suspicious): {inv['reasoning']}"))

    if results["address"]["outcome"] == "needs_resubmission":
        issues.append(("address", "conditional", "address mismatch: " + results["address"]["detail"]))

    if len(issues) == 0:
        return {"persona_id": persona_id, "verdict": "APPROVE", "reasons": ["all checks clean"]}

    if len(issues) >= 2:
        return {
            "persona_id": persona_id,
            "verdict": "REJECT",
            "reasons": [reason for (_, _, reason) in issues],
        }

    _, severity, reason = issues[0]
    verdict_map = {"reject": "REJECT", "warning": "WARNING_RESUBMISSION", "conditional": "CONDITIONAL_APPROVAL"}
    return {
        "persona_id": persona_id,
        "verdict": verdict_map[severity],
        "reasons": [reason],
    }


def process_persona(persona_id: str) -> dict:
    results = run_all_checks(persona_id)
    investigations = run_investigations(persona_id, results)
    verdict = build_verdict(persona_id, results, investigations)
    return {"results": results, "investigations": investigations, "verdict": verdict}


if __name__ == "__main__":
    print("precomputing recycling detection across the applicant pool...")
    get_recycling_results()

    batch_results = []
    for persona in tqdm(personas, desc="personas", unit="persona"):
        persona_id = persona["persona_id"]
        print(f"\nprocessing {persona_id}...")
        outcome = process_persona(persona_id)
        batch_results.append({
            "persona_id": persona_id,
            "scenario": persona["scenario"],
            "verdict": outcome["verdict"]["verdict"],
            "reasons": outcome["verdict"]["reasons"],
        })

    print("\n" + "=" * 100)
    print(f"{'persona_id':35s} {'scenario':30s} {'verdict':22s}")
    print("-" * 100)
    for r in batch_results:
        print(f"{r['persona_id']:35s} {r['scenario']:30s} {r['verdict']:22s}")
    print("=" * 100)

    print("\nDetailed reasons for anything not APPROVE:")
    for r in batch_results:
        if r["verdict"] != "APPROVE":
            print(f"\n{r['persona_id']} ({r['scenario']}) -- {r['verdict']}")
            for reason in r["reasons"]:
                print(f"  - {reason}")
