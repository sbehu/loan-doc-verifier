"""
Balance reconciliation check.
Extracts transaction rows from bank_statement.png via Textract's table
detection (see textract_bank.py for why this document type gets OCR
table extraction while UPI statements stay on the vision-LLM approach),
then reconciliation math (previous balance + amount == this row's
balance) is done in plain Python, not by any model.

No confidence banding needed here -- reconciliation is exact
arithmetic, so this stays a simple flagged: bool like the original.
"""

from pathlib import Path

from dotenv import load_dotenv

from checks.textract_bank import extract_bank_statement_rows

base_dir = Path(__file__).resolve().parent.parent
load_dotenv(base_dir / ".env")


def check_balance(persona_id: str) -> dict:
    statement_path = base_dir / "data" / "documents" / persona_id / "bank_statement.png"
    if not statement_path.exists():
        return {
            "persona_id": persona_id,
            "check": "balance_reconciliation",
            "flagged": False,
            "detail": "no bank statement present -- check not applicable",
        }

    rows = extract_bank_statement_rows(persona_id)

    if len(rows) < 2:
        return {
            "persona_id": persona_id,
            "check": "balance_reconciliation",
            "flagged": False,
            "detail": f"only {len(rows)} row(s) extracted -- not enough to reconcile",
        }

    # Reconciliation is the primary signal, not any single transcribed
    # field. If a row doesn't reconcile at first glance, check whether its
    # balance already occurred earlier in the sequence -- balance is a
    # running cumulative total, so a genuine repeat essentially never
    # happens naturally. A repeat almost always means this is the same
    # physical row read twice (chunk overlap), with some field (often
    # date) transcribed inconsistently between the two reads -- not real
    # tampering. Skip it without moving the running balance forward.
    # Only flag as tampering if the mismatch can't be explained this way.
    last_valid_balance = float(rows[0]["balance"])
    seen_balances = {last_valid_balance}
    checked = 1

    for i in range(1, len(rows)):
        amount = float(rows[i]["amount"])
        this_balance = float(rows[i]["balance"])
        expected = last_valid_balance + amount

        if abs(expected - this_balance) <= 0.5:  # small tolerance for rounding
            last_valid_balance = this_balance
            seen_balances.add(this_balance)
            checked += 1
            continue

        if this_balance in seen_balances:
            continue  # duplicate artifact from chunk overlap -- skip

        return {
            "persona_id": persona_id,
            "check": "balance_reconciliation",
            "flagged": True,
            "detail": (
                f"transaction on {rows[i]['date']} ({rows[i]['description']}) does not "
                f"reconcile: expected balance {expected:.2f}, printed balance {this_balance:.2f}"
            ),
        }

    skipped = len(rows) - checked
    detail = f"all {checked} rows reconcile correctly"
    if skipped:
        detail += f" ({skipped} duplicate artifact(s) skipped)"
    return {
        "persona_id": persona_id,
        "check": "balance_reconciliation",
        "flagged": False,
        "detail": detail,
    }


if __name__ == "__main__":
    result = check_balance("P003_doc_tamper_careless")
    print(result)
