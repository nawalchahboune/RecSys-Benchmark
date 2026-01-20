from pathlib import Path
from typing import Optional
from item_encoder import ClueWebSeqDataset
from torch.utils.data import DataLoader

def get_dataloader(
    input_path: str,
    target_path: Optional[str],
    max_seq_len: int = 50,
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

if __name__ == "__main__":
    data_dir = Path("../data/ClueWeb-Reco/ordered_id_splits")
    valid_input = data_dir / "valid_input.tsv"
    valid_target = data_dir / "valid_target.tsv"

    valid_loader = get_dataloader(
        input_path=str(valid_input),
        target_path=str(valid_target),
        max_seq_len=50,
        batch_size=32,
        shuffle=False,  # en eval on ne shuffle pas
    )

    for batch in valid_loader:
        seqs, lengths, targets = batch  # seqs: [B, max_seq_len]
        print("seqs shape:", seqs.shape)
        print("lengths shape:", lengths.shape)
        print("targets shape:", targets.shape)
        break

