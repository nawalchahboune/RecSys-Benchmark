from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Set


def iter_sequences(path: Path) -> Iterable[List[int]]:
    """
    Iterate over sequences of item IDs from a TSV file.
    """
    with path.open("r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue

            # skip header
            if first and "session_id" in line:
                first = False
                continue
            first = False

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            seq_str = parts[1].strip()
            if not seq_str:
                continue

            try:
                yield [int(tok) for tok in seq_str.split(",") if tok]
            except ValueError:
                # skip malformed lines
                continue


def collect_seen_ids(files: List[Path]) -> Set[int]:
    seen: Set[int] = set()
    for p in files:
        if not p.exists():
            continue
        for seq in iter_sequences(p):
            seen.update(seq)
    return seen


def main():
    data_dir = Path("../data/ClueWeb-Reco/ordered_id_splits")  # ClueWeb-Reco split files 

    candidate_files = [
        data_dir / "train_input.tsv",
        data_dir / "valid_input.tsv",
        data_dir / "test_input.tsv",
    ]
    candidate_files = [p for p in candidate_files if p.exists()]

    if not candidate_files:
        raise RuntimeError(f"No split files found in {data_dir}")

    seen = collect_seen_ids(candidate_files)

    out_path = data_dir / "seen_item_ids.txt"
    out_path.write_text("\n".join(map(str, sorted(seen))), encoding="utf-8")

    print("Read files:")
    for p in candidate_files:
        print(" -", p.name)

    print("Unique seen item IDs:", len(seen))
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
