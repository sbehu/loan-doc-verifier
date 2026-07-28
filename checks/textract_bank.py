"""
Bank statement table extraction via AWS Textract.

Bank statements (unlike UPI statements, which vary across gpay/phonepe/
paytm UI layouts) are genuine tables -- a header row plus a consistent
grid of columns. That's exactly what Textract's table-detection feature
is built for, so this replaces the old vision-LLM chunking approach for
bank_statement.png specifically. income_upi.py's UPI-statement reading
stays on the vision-LLM approach, since that document type is a
free-form styled list, not a table -- a different tool for a different
document shape, not inconsistency.

Column meaning is resolved by matching each detected header cell's own
text against keywords ("withdraw"/"debit", "deposit"/"credit",
"balance", "date", "narration"/"description"/"particulars") rather than
hardcoding column positions or count -- so this isn't tied to our one
synthetic template's exact 5-column layout. It should hold up against
real bank statements with different column counts/order/wording, as
long as they use standard banking terminology (which is the honest,
not-100%-guaranteed assumption here).

Textract's synchronous API caps documents at 10,000px on a side, so
very tall statement images still need to be split -- but into at most
a couple of large (~9000px) pieces, not ~20 small 900px slices like the
vision-LLM approach needed. Header/column meaning is only resolved once,
from the first chunk (which reliably contains the real header row);
later chunks reuse that same column mapping, since they're a
continuation of the same table with no header of their own.
"""

import re
from io import BytesIO
from pathlib import Path

import boto3
from dotenv import load_dotenv
from PIL import Image

base_dir = Path(__file__).resolve().parent.parent
load_dotenv(base_dir / ".env")

textract = boto3.client("textract")

MAX_SIDE = 4000  # Textract's own documented limit is 10,000px, but testing
                  # showed it silently returns almost nothing (no TABLE
                  # blocks, near-zero WORD blocks) well before that on tall,
                  # dense statement images -- 2000/4000px crops both worked
                  # correctly (word/cell counts scaled proportionally), 9000px
                  # collapsed to near-empty. Staying well under that cliff.
OVERLAP = 500

EDGE_MARGIN_PX = 150  # any row whose bounding box falls within this many
                       # pixels of an INTERNAL chunk boundary gets discarded.
                       # Confirmed directly on a real (non-synthetic) bank
                       # statement: a multi-line row got cut mid-narration by
                       # an arbitrary chunk slice, and the leftover fragment
                       # was wrongly stitched onto the following row,
                       # corrupting its amount/balance. The overlapping
                       # neighboring chunk already captures the same row
                       # fully, away from its own edges -- so discarding the
                       # edge-adjacent copy loses nothing.

DATE_KEYWORDS = ["date"]
DESC_KEYWORDS = ["narration", "description", "particulars"]
WITHDRAWAL_KEYWORDS = ["withdraw", "debit"]
DEPOSIT_KEYWORDS = ["deposit", "credit"]
BALANCE_KEYWORDS = ["balance"]


def _clean_number(text):
    if not text:
        return None
    cleaned = re.sub(r"[^\d.\-]", "", text)
    if not cleaned or cleaned in ("-", "."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


# Matches the actual date token within a cell's text, discarding any
# secondary label sharing the same cell (e.g. "2026-04-06 WDL TFR" -- a
# transaction-type marker printed under the date in the same column).
# Covers a few common formats so this isn't hardcoded to our one
# synthetic template's YYYY-MM-DD style; falls back to the raw text if
# nothing matches, rather than silently dropping the field.
DATE_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}"
)


def _clean_date(text):
    if not text:
        return text
    match = DATE_PATTERN.search(text)
    return match.group(0) if match else text.strip()


def _match_column(headers: dict, keywords: list):
    for col_idx, text in headers.items():
        lower = (text or "").lower()
        if any(kw in lower for kw in keywords):
            return col_idx
    return None


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _cells_from_blocks(blocks: list) -> list:
    """Returns the cells of whichever detected TABLE has the most cells
    (the real transaction table), as {row, col, text} dicts -- NOT a
    flattened merge of every table on the page. This statement image has
    two table-like regions: a small account-summary info block near the
    top, and the actual transaction table. Both number their own rows
    starting from 1, so blindly merging them by (row, col) would corrupt
    both -- picking the largest table avoids that collision entirely."""
    block_map = {b["Id"]: b for b in blocks}
    tables = []  # list of cell-lists, one per detected TABLE block
    for block in blocks:
        if block["BlockType"] != "TABLE":
            continue
        table_cells = []
        for rel in block.get("Relationships", []):
            if rel["Type"] != "CHILD":
                continue
            for cell_id in rel["Ids"]:
                cell = block_map.get(cell_id)
                if not cell or cell["BlockType"] != "CELL":
                    continue
                text_parts = []
                for cell_rel in cell.get("Relationships", []):
                    if cell_rel["Type"] != "CHILD":
                        continue
                    for word_id in cell_rel["Ids"]:
                        word = block_map.get(word_id)
                        if word and word["BlockType"] in ("WORD", "SELECTION_ELEMENT"):
                            text_parts.append(word.get("Text", ""))
                bbox = cell["Geometry"]["BoundingBox"]
                table_cells.append({
                    "row": cell["RowIndex"],
                    "col": cell["ColumnIndex"],
                    "text": " ".join(text_parts).strip(),
                    "top": bbox["Top"],
                    "bottom": bbox["Top"] + bbox["Height"],
                })
        if table_cells:
            tables.append(table_cells)

    if not tables:
        return []
    return max(tables, key=len)


def _find_col_map(grid: dict):
    """Scans the first several rows for whichever one is the real column
    header -- i.e. the row where date/withdrawal/deposit/balance
    keywords each match a DIFFERENT column. Doesn't assume row 1 is the
    header: this statement's account-summary info block ("Statement
    Period: ... Opening Balance: Rs. ...") gets detected as row 1 of the
    SAME table, ahead of the real header row. Rows that merely contain
    one matching word mixed into unrelated text won't have 3+ distinct
    matched columns, so they're skipped in favor of the real header."""
    for row_idx in sorted(grid.keys())[:6]:
        headers = grid[row_idx]
        col_map = {
            "date": _match_column(headers, DATE_KEYWORDS),
            "description": _match_column(headers, DESC_KEYWORDS),
            "withdrawal": _match_column(headers, WITHDRAWAL_KEYWORDS),
            "deposit": _match_column(headers, DEPOSIT_KEYWORDS),
            "balance": _match_column(headers, BALANCE_KEYWORDS),
        }
        key_cols = [
            v for v in (col_map["date"], col_map["withdrawal"], col_map["deposit"], col_map["balance"])
            if v is not None
        ]
        if len(key_cols) >= 3 and len(set(key_cols)) == len(key_cols):
            return col_map
    return None


def extract_bank_statement_rows(persona_id: str) -> list:
    """Returns a list of {date, description, amount, balance} dicts, one
    per transaction row, extracted via Textract's table detection."""
    statement_path = base_dir / "data" / "documents" / persona_id / "bank_statement.png"
    img = Image.open(statement_path).convert("RGB")
    w, h = img.size

    if h <= MAX_SIDE:
        chunk_starts = [0]
    else:
        stride = MAX_SIDE - OVERLAP
        chunk_starts = list(range(0, h, stride))

    col_map = None
    all_rows = []
    seen = set()

    is_last_chunk_idx = len(chunk_starts) - 1

    for i, y in enumerate(chunk_starts):
        crop = img.crop((0, y, w, min(y + MAX_SIDE, h)))
        crop_height = crop.height
        resp = textract.analyze_document(
            Document={"Bytes": _to_png_bytes(crop)}, FeatureTypes=["TABLES"]
        )
        cells = _cells_from_blocks(resp["Blocks"])

        grid = {}
        row_extent = {}  # row_idx -> [min_top, max_bottom], normalized 0-1 within this crop
        for c in cells:
            grid.setdefault(c["row"], {})[c["col"]] = c["text"]
            extent = row_extent.setdefault(c["row"], [c["top"], c["bottom"]])
            extent[0] = min(extent[0], c["top"])
            extent[1] = max(extent[1], c["bottom"])
        if not grid:
            continue

        if i == 0:
            col_map = _find_col_map(grid)
            if col_map is None:
                raise ValueError(
                    f"could not find a real header row in {persona_id}'s bank "
                    "statement table -- table detection or header wording issue"
                )

        for row_idx in sorted(grid.keys()):
            row = grid[row_idx]

            if row_idx in row_extent:
                top_px = row_extent[row_idx][0] * crop_height
                bottom_px = row_extent[row_idx][1] * crop_height
                near_top_edge = top_px < EDGE_MARGIN_PX
                near_bottom_edge = (crop_height - bottom_px) < EDGE_MARGIN_PX
                if near_top_edge and i != 0:
                    continue  # previous chunk's overlap already captured this row intact
                if near_bottom_edge and i != is_last_chunk_idx:
                    continue  # next chunk's overlap will capture this row intact

            balance = _clean_number(row.get(col_map["balance"]))
            if balance is None:
                continue

            withdrawal = (
                _clean_number(row.get(col_map["withdrawal"]))
                if col_map["withdrawal"] is not None else None
            )
            deposit = (
                _clean_number(row.get(col_map["deposit"]))
                if col_map["deposit"] is not None else None
            )
            if withdrawal:
                amount = -withdrawal
            elif deposit:
                amount = deposit
            else:
                continue  # neither column had a value -- not a transaction row

            date = _clean_date(row.get(col_map["date"])) if col_map["date"] is not None else None
            description = (
                row.get(col_map["description"], "")
                if col_map["description"] is not None else ""
            )

            key = (date, amount, balance)
            if key in seen:
                continue  # duplicate row from chunk overlap
            seen.add(key)
            all_rows.append({
                "date": date, "description": description,
                "amount": amount, "balance": balance,
            })

    return all_rows


if __name__ == "__main__":
    rows = extract_bank_statement_rows("P001_clean_salaried")
    print(f"extracted {len(rows)} rows")
    for r in rows[:5]:
        print(r)
