from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader

class ClueWebSeqDataset(Dataset):
    def __init__(
        self,
        input_path: str,
        target_path: Optional[str] = None,
        max_seq_len: int = 50,
    ):
        """
        input_path: chemin vers valid_input.tsv (ou train_input.tsv)
        target_path: chemin vers valid_target.tsv (ou train_target.tsv), ou None pour le test
        max_seq_len: longueur maximale de séquence (on tronque/pad)
        """
        self.input_path = Path(input_path)
        self.target_path = Path(target_path) if target_path is not None else None
        self.max_seq_len = max_seq_len

        # Charge tout en mémoire
        self.sequences: List[List[int]] = []
        self.targets: Optional[List[int]] = None

        self._load_sequences()
        if self.target_path is not None:
            self._load_targets()
            assert len(self.sequences) == len(self.targets), "Inputs/targets pas alignés"

    def _load_sequences(self):
        with self.input_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                # Skip header row
                if i == 0:
                    continue
                # TSV format: session_id \t comma-separated item IDs
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                item_ids = [int(tok) for tok in parts[1].split(',')]
                self.sequences.append(item_ids)

    def _load_targets(self):
        self.targets = []
        with self.target_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                # Skip header row
                if i == 0:
                    continue
                # TSV format: session_id \t target_id
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                self.targets.append(int(parts[1]))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Optional[int]]:
        """
        Retourne:
          - seq_tensor: LongTensor de shape [max_seq_len] (avec padding à gauche)
          - length: longueur réelle (sans padding)
          - target: int (ou None si split de test)
        """
        seq = self.sequences[idx]

        # Tronquer aux max_seq_len derniers items
        if len(seq) > self.max_seq_len:
            seq = seq[-self.max_seq_len:]

        length = len(seq)

        # Padding à gauche avec 0
        seq_tensor = torch.zeros(self.max_seq_len, dtype=torch.long)
        seq_tensor[-length:] = torch.tensor(seq, dtype=torch.long)

        target = None
        if self.targets is not None:
            target = self.targets[idx]

        return seq_tensor, length, target
    
