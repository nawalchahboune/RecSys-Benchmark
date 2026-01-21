from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import DataLoader

from item_encoder import ClueWebSeqDataset

import struct
from typing import List

def load_seen_items(path: Path) -> List[int]:
    """Load the list of seen items"""
    with path.open("r") as f:
        return [int(line.strip()) for line in f if line.strip()]
    
def load_sequences_from_dataset(dataset: ClueWebSeqDataset) -> List[List[int]]:
    """Extract sequences from ClueWebSeqDataset (reuses existing class)"""
    return dataset.sequences

def load_item_text_simple(seen_items: List[int]) -> Dict[int, str]:
    # """Simple version: use item IDs as text (for prototyping)"""
    # return {item_id: f"item_{item_id}" for item_id in seen_items}
    """Load item from the cwid_to_id.tsv file"""
    item_text = {}
    seen_set = set(seen_items)
    data_dir = Path("../data/ClueWeb-Reco")
    cwid_to_id_path = data_dir / "cwid_to_id.tsv"
    
    print(f"Loading item text from {cwid_to_id_path}...")
    with cwid_to_id_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            # parts[0] = cwid (string like "clueweb22-en0014-00-00001")
            # parts[1] = internal item_id (int)
            cwid = parts[0]  # Keep as string
            item_id = int(parts[1])
            if item_id in seen_set:
                item_text[item_id] = cwid  # Use the actual cwid as text
    return item_text


def get_dataloader(
    input_path: str,
    target_path: Optional[str],
    max_seq_len: int = 300,
    batch_size: int = 128,
    shuffle: bool = True,
):
    dataset = ClueWebSeqDataset(
        input_path=input_path,
        target_path=target_path,
        max_seq_len=max_seq_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader

def save_orbit_bin(preds: List[List[int]], K: int, out_path: str) -> None:
    """
    preds: list of sessions; each is a list (len>=K) of cw_internal_id (int).
    K:     number of items to keep per session (top-K).
    out_path: where to write the .bin file.

    Format:
      <4 bytes uint32: num_sessions>
      <4 bytes uint32: K>
      <num_sessions * K * int32: predicted cw_internal_id>
    """
    num_sessions = len(preds)

    with open(out_path, "wb") as f:
        # write header
        f.write(struct.pack("<I", num_sessions))  # num_sessions
        f.write(struct.pack("<I", K))             # K

        for row in preds:
            # ensure length >= K, pad with 0 if needed
            if len(row) < K:
                row = row + [0] * (K - len(row))
            else:
                row = row[:K]

            # write K int32
            for x in row:
                f.write(struct.pack("<i", int(x)))

    
