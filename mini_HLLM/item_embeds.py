from sentence_transformers import SentenceTransformer
from pathlib import Path
import csv
import numpy as np

from helpers import load_seen_items  # must return List[int] or Set[int]


def load_item_texts_from_cwid_to_id(cwid_to_id_path: Path, seen_items: set[int]) -> dict[int, str]:
    """
    Minimal loader: read the mapping file and build item_id -> text.
    IMPORTANT: adapt the column names once you inspect the header.
    """
    item_text = {}

    with cwid_to_id_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # Print header once to verify field names
        print("Columns:", reader.fieldnames)

        for row in reader:
            try:
                item_id = int(row.get("cw_internal_id") or row.get("internal_id") or row.get("item_id"))
            except Exception:
                continue

            if item_id not in seen_items:
                continue

            # Try to build a short text (URL is often available; replace with title/snippet if you have them)
            url = (row.get("url") or "").strip()
            cwid = (row.get("cwid") or row.get("clueweb_id") or "").strip()

            # Fallback text if you don't have real page content
            text = url if url else cwid
            if not text:
                continue

            item_text[item_id] = text

    return item_text


def main():
    model = SentenceTransformer("finetuned_item_embedder")  # checkpoint path

    data_dir = Path("../data/ClueWeb-Reco/ordered_id_splits")

    # 1) seen items
    seen_items = set(load_seen_items(data_dir / "seen_item_ids.txt"))
    print(f"Loaded {len(seen_items)} seen items")

    # 2) load texts for those items
    cwid_to_id_path = Path("../data/ClueWeb-Reco/cwid_to_id.tsv")
    item_text = load_item_texts_from_cwid_to_id(cwid_to_id_path, seen_items)
    print(f"Loaded text for {len(item_text)} items")

    # 3) align ids + texts
    item_ids = sorted(item_text.keys())
    texts = [item_text[i] for i in item_ids]

    # 4) memory/time controls
    # limit max tokens processed by the model (long texts are truncated anyway) 
    model.max_seq_length = min(getattr(model, "max_seq_length", 256), 128)

    # 5) encode in batches; normalize for cosine-as-dot-product 
    embs = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # 6) save: keep item_ids + embedding matrix
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "item_ids.npy", np.array(item_ids, dtype=np.int64))
    np.save(out_dir / "item_embs.npy", embs.astype(np.float32))
    print("Saved:", out_dir / "item_ids.npy")
    print("Saved:", out_dir / "item_embs.npy")
    print("Embeddings shape:", embs.shape)


if __name__ == "__main__":
    main()
