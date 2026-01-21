from sentence_transformers import SentenceTransformer
from pathlib import Path
import csv
import numpy as np

from helpers import load_seen_items  # must return List[int] or Set[int]


def load_item_texts_from_cwid_to_id(cwid_to_id_path: Path, seen_items: set[int]) -> dict[int, str]:
    """
    Minimal loader: read the mapping file and build item_id -> text.
    Format: cwid \t internal_id (no header)
    """
    item_text = {}

    with cwid_to_id_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            try:
                cwid = parts[0]  # e.g., "clueweb22-en0014-00-00001"
                item_id = int(parts[1])  # internal ID
            except (ValueError, IndexError):
                continue

            if item_id not in seen_items:
                continue

            # Use cwid as text representation
            item_text[item_id] = cwid

    return item_text


def main():
    model = SentenceTransformer("checkpoints/item_embedder")  # checkpoint path

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

    np.save(out_dir / "item_ids_30.npy", np.array(item_ids, dtype=np.int64))
    np.save(out_dir / "item_embs_30.npy", embs.astype(np.float32))
    print("Saved:", out_dir / "item_ids_30.npy")
    print("Saved:", out_dir / "item_embs_30.npy")
    print("Embeddings shape:", embs.shape)


if __name__ == "__main__":
    main()
