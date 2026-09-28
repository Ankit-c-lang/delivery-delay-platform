"""Check that the nine Olist CSVs are present, and explain how to get them if not.

This script deliberately does **not** download anything. The dataset is CC BY-NC-SA and
sits behind Kaggle's terms; scraping it or committing it would breach both. Acquisition is
a documented manual step, and this script's only job is to make a missing or partial
download fail immediately with instructions, rather than halfway through an ETL run.

Run with::

    python -m scripts.download_data
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.etl.load_raw import SPECS, csv_header, raw_data_dir

KAGGLE_DATASET = "olistbr/brazilian-ecommerce"

INSTRUCTIONS = f"""
How to get the data
-------------------

The dataset is CC BY-NC-SA licensed and is never committed to this repository.

Option 1 — Kaggle CLI (needs ~/.kaggle/kaggle.json, mode 600):

    pip install kaggle
    kaggle datasets download -d {KAGGLE_DATASET} -p data/raw --unzip

Option 2 — browser:

    1. Open https://www.kaggle.com/datasets/{KAGGLE_DATASET}
    2. Download the archive and unzip it into data/raw/
    3. Do not rename the files; the loader matches them by name.

Then re-run this script. `data/` is gitignored — keep it that way.
""".strip()


def check(data_dir: Path | None = None) -> list[str]:
    """Report what is wrong with the raw data directory.

    Args:
        data_dir: Directory to check. Defaults to ``paths.raw_data`` in
            ``configs/base.yaml``.

    Returns:
        Human-readable problems, empty when every file is present and its header matches
        the loader's spec. A header check is included because a truncated or wrong-dataset
        download is otherwise indistinguishable from a good one until ``COPY`` fails.
    """
    data_dir = data_dir or raw_data_dir()
    problems: list[str] = []

    if not data_dir.is_dir():
        return [f"{data_dir}/ does not exist"]

    for spec in SPECS:
        path = data_dir / spec.csv_name
        if not path.exists():
            problems.append(f"missing: {path}")
            continue
        if path.stat().st_size == 0:
            problems.append(f"empty: {path}")
            continue
        try:
            header = csv_header(path)
        except (OSError, UnicodeDecodeError, StopIteration) as exc:
            problems.append(f"unreadable: {path} ({exc})")
            continue
        if header != spec.column_names:
            problems.append(
                f"header mismatch: {path.name}\n"
                f"    expected {spec.column_names}\n"
                f"    found    {header}"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    """Exit 0 when the data is ready, 1 otherwise."""
    argv = sys.argv[1:] if argv is None else argv
    data_dir = Path(argv[0]) if argv else None
    problems = check(data_dir)

    resolved = data_dir or raw_data_dir()
    if problems:
        print(f"Raw data in {resolved}/ is not ready:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(f"\n{INSTRUCTIONS}", file=sys.stderr)
        return 1

    total_mb = sum((resolved / s.csv_name).stat().st_size for s in SPECS) / 1e6
    print(f"All {len(SPECS)} Olist CSVs present in {resolved}/ ({total_mb:.0f} MB), headers match.")
    print("Next: python -m src.etl.load_raw")
    return 0


if __name__ == "__main__":
    sys.exit(main())
