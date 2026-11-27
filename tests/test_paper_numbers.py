"""Every table cell printed in the paper against the tables built from results/.

tests/paper_numbers.csv was typed from the camera-ready PDF.
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_tables import build_tables  # noqa: E402

# Cells where the paper rounded twice (four decimals, then three). See docs/DIFFERENCES.md.
KNOWN_DIFFERENCES = {
    ("table17_position_fusion", "medmcqa", "Llama-3.1-8B-Instruct"): "0.828 / 0.850 / 0.865",
    ("table17_position_fusion", "medqa_usmle", "Llama-3.1-8B-Instruct"): "0.774 / 0.779 / 0.798",
    ("table17_position_fusion", "pubmedqa_labeled", "Meditron-7B"): "0.784 / 0.803 / 0.823",
}


def cell(table, row, column):
    if isinstance(table.index[0], tuple):
        row = tuple(row.split("|"))
    return str(table.loc[row, column])


def test_tables_match_paper():
    tables = build_tables(ROOT / "results")
    with open(ROOT / "tests" / "paper_numbers.csv") as f:
        paper = list(csv.DictReader(f))
    assert len(paper) == 343
    mismatches = []
    for entry in paper:
        key = (entry["table"], entry["row"], entry["column"])
        expected = KNOWN_DIFFERENCES.get(key, entry["paper_value"])
        built = cell(tables[entry["table"]], entry["row"], entry["column"])
        if built != expected:
            mismatches.append((key, entry["paper_value"], built))
    assert not mismatches, mismatches
