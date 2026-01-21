from __future__ import annotations

from collections import Counter, defaultdict
from urllib.parse import urlparse
import csv

from helpers import load_item_text_simple

import math
import struct
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from helpers import save_orbit_bin

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# ---------- IO (ordered_id_splits) ----------
def read_ordered_tsv(path: Path) -> List[List[int]]:
    seqs: List[List[int]] = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            hist = parts[1].strip()
            if not hist:
                seqs.append([])
                continue
            if "," in hist:
                seq = [int(x) for x in hist.split(",") if x.strip()]
            else:
                seq = [int(x) for x in hist.split() if x.strip()]
            seqs.append(seq)
    return seqs


def read_valid_targets(path: Path) -> List[int]:
    tgts: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            tgts.append(int(parts[1]))
    return tgts


# ---------- Submission writer ----------
def write_bin(out_path: str, all_preds: List[List[int]], K: int) -> None:
    n = len(all_preds)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(struct.pack("<I", K))
        for row in all_preds:
            if len(row) != K:
                raise ValueError(f"Row len={len(row)} expected K={K}")
            for x in row:
                f.write(struct.pack("<i", int(x)))


# ---------- Dataset: (prefix -> next) pairs ----------
class NextItemDataset(Dataset):
    def __init__(
        self,
        seqs: List[List[int]],
        id2idx: Dict[int, int],
        min_prefix_len: int = 1,
        max_prefix_len: int = 20,
        max_pairs: Optional[int] = None,
    ):
        """
        Builds training pairs from sequences:
          prefix = seq[:t], target = seq[t]
        We keep only IDs present in id2idx (embed table).
        """
        self.examples: List[Tuple[List[int], int]] = []

        for seq in seqs:
            seq = [i for i in seq if i in id2idx]
            if len(seq) < 2:
                continue

            # generate multiple pairs per sequence
            for t in range(1, len(seq)):
                prefix = seq[max(0, t - max_prefix_len):t]
                if len(prefix) < min_prefix_len:
                    continue
                target = seq[t]
                self.examples.append((prefix, target))

                if max_pairs is not None and len(self.examples) >= max_pairs:
                    break
            if max_pairs is not None and len(self.examples) >= max_pairs:
                break

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        return self.examples[idx]


def collate_prefix_batch(batch: List[Tuple[List[int], int]], id2idx: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      x_idx: LongTensor [B, T]  (item indices, padded with 0)
      lengths: LongTensor [B]
      y_idx: LongTensor [B]     (target item index)
    """
    prefixes, targets = zip(*batch)
    lengths = torch.tensor([len(p) for p in prefixes], dtype=torch.long)
    T = int(lengths.max().item())
    B = len(prefixes)

    x_idx = torch.zeros((B, T), dtype=torch.long)
    for i, p in enumerate(prefixes):
        idxs = [id2idx[it] for it in p]
        x_idx[i, -len(idxs):] = torch.tensor(idxs, dtype=torch.long)

    y_idx = torch.tensor([id2idx[t] for t in targets], dtype=torch.long)
    return x_idx, lengths, y_idx


# ---------- Model: GRU user encoder ----------
class UserGRU(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x_emb: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x_emb: [B, T, D]
        lengths: [B]
        returns: predicted next embedding [B, D] (L2-normalized)
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x_emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)          # h: [1, B, H]
        h = h.squeeze(0)                # [B, H]
        z = self.proj(h)                # [B, D]
        z = nn.functional.normalize(z, dim=-1)
        return z


# ---------- InfoNCE (in-batch negatives) ----------
def info_nce_inbatch(query: torch.Tensor, pos: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    query: [B, D] predicted embedding
    pos:   [B, D] true next-item embedding
    Uses other positives in the batch as negatives. [web:274]
    """
    query = nn.functional.normalize(query, dim=-1)
    pos = nn.functional.normalize(pos, dim=-1)
    logits = (query @ pos.T) / temperature   # [B, B]
    labels = torch.arange(logits.size(0), device=logits.device)
    return nn.functional.cross_entropy(logits, labels)


# ---------- Evaluation (Recall@K, NDCG@K for one target) ----------
def recall_ndcg_at_k(targets: List[int], preds: List[List[int]], K: int) -> Tuple[float, float]:
    hits = 0
    ndcg = 0.0
    n = len(targets)
    for t, row in zip(targets, preds):
        topk = row[:K]
        if t in topk:
            hits += 1
            rank = topk.index(t) + 1
            ndcg += 1.0 / math.log2(rank + 1)
    return hits / n, ndcg / n

def compute_gru_session_embeddings(
    model: nn.Module,
    valid_seqs: List[List[int]],
    id2idx: Dict[int, int],
    item_ids: np.ndarray,
    item_embs: torch.Tensor,
    device: str,
    max_prefix_len: int = 20,
) -> np.ndarray:
    """
    Retourne U: [N_sessions, D] embeddings de session (u) sortis par le GRU.
    Pour les sessions vides (après filtrage id2idx), met un vecteur 0.
    """
    model.eval()
    D = item_embs.shape[1]
    U = []

    with torch.no_grad():
        for seq in valid_seqs:
            prefix = [i for i in seq if i in id2idx][-max_prefix_len:]
            if len(prefix) == 0:
                U.append(np.zeros(D, dtype=np.float32))
                continue

            x = torch.tensor([[id2idx[i] for i in prefix]], dtype=torch.long, device=device)
            lengths = torch.tensor([x.shape[1]], dtype=torch.long, device=device)
            x_emb = item_embs[x]  # [1,T,D]

            u = model(x_emb, lengths).squeeze(0)  # [D], déjà normalisé
            U.append(u.detach().cpu().numpy().astype(np.float32))

    return np.stack(U, axis=0)


def plot_tsne_clusters(
    U: np.ndarray,
    labels: np.ndarray,
    out_path: str = "gru_user_tsne.png",
    max_points: int = 2000,
):
    """
    U: [N,D] session embeddings
    labels: [N] cluster id
    """
    N = U.shape[0]
    idx = np.arange(N)
    if N > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=max_points, replace=False)

    U_sub = U[idx]
    labels_sub = labels[idx]

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init="pca", random_state=0)
    coords = tsne.fit_transform(U_sub)

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels_sub, s=10, cmap="tab10", alpha=0.75)
    plt.colorbar(label="Cluster ID")
    plt.title("GRU session embeddings (t-SNE)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)


def summarize_clusters(
    labels: np.ndarray,
    tgts: List[int],
    preds_at_100: List[List[int]],
    out_csv: str = "gru_cluster_summary.csv",
):
    """
    Petit résumé interprétable par cluster:
    - nb de sessions
    - hit@100 (sur validation)
    """
    import csv
    from collections import defaultdict

    by_c = defaultdict(list)
    for i, c in enumerate(labels):
        by_c[int(c)].append(i)

    rows = []
    for c, idxs in sorted(by_c.items()):
        n = len(idxs)
        hits = 0
        for i in idxs:
            if tgts[i] in preds_at_100[i]:
                hits += 1
        rows.append({"cluster": c, "n_sessions": n, "hit@100": hits / n if n else 0.0})

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cluster", "n_sessions", "hit@100"])
        w.writeheader()
        w.writerows(rows)

    print("Saved:", out_csv)

def export_cluster_cwids(
    labels: np.ndarray,
    valid_seqs: List[List[int]],
    id2cwid: Dict[int, str],   # from load_item_text_simple
    out_csv: str = "cluster_cwids.csv",
):
    by_c = defaultdict(list)
    for i, c in enumerate(labels):
        by_c[int(c)].append(i)

    rows = []
    for c, idxs in sorted(by_c.items()):
        cwids = set()
        for i in idxs:
            for it in valid_seqs[i]:
                cwid = id2cwid.get(int(it))
                if cwid:
                    cwids.add(cwid)
        rows.append({"cluster": c, "cwids": ";".join(sorted(cwids))})

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cluster", "cwids"])
        w.writeheader()
        w.writerows(rows)

    print("Saved:", out_csv)

def main():
    device = "cpu"  # or "mps" if you want Apple GPU
    base = Path("../data/ClueWeb-Reco/ordered_id_splits")
    artifacts = Path("artifacts")

    # Load item embedding table (frozen)
    item_ids = np.load(artifacts / "item_ids_30.npy").astype(np.int64)     # (N,)
    item_embs_np = np.load(artifacts / "item_embs_30.npy").astype(np.float32)  # (N,D)
    N, D = item_embs_np.shape

    # map cw_internal_id -> row index in item_embs
    id2idx = {int(i): idx for idx, i in enumerate(item_ids)}

    # Put embeddings in torch (freeze)
    item_embs = torch.from_numpy(item_embs_np).to(device)  # [N, D]

    # -------- data --------
    train_path = base / "train_input.tsv"
    if not train_path.exists():
        # fallback (not ideal, but works to debug)
        train_path = base / "valid_input.tsv"

    train_seqs = read_ordered_tsv(train_path)
    valid_seqs = read_ordered_tsv(base / "valid_input.tsv")
    valid_tgts = read_valid_targets(base / "valid_target.tsv")

    # Training pairs
    train_ds = NextItemDataset(train_seqs, id2idx, min_prefix_len=1, max_prefix_len=20, max_pairs=200_000)
    print("Train pairs:", len(train_ds))

    def collate_fn(batch):
        return collate_prefix_batch(batch, id2idx)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn)

    # -------- model --------
    model = UserGRU(emb_dim=D, hidden_dim=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    # -------- training (1-3 epochs CPU friendly) --------
    model.train()
    for epoch in range(1, 3):
        total = 0.0
        for x_idx, lengths, y_idx in train_loader:
            x_idx = x_idx.to(device)
            lengths = lengths.to(device)
            y_idx = y_idx.to(device)

            x_emb = item_embs[x_idx]          # [B,T,D] lookup in frozen table
            pos_emb = item_embs[y_idx]        # [B,D]

            pred_emb = model(x_emb, lengths)  # [B,D]
            loss = info_nce_inbatch(pred_emb, pos_emb, temperature=0.07)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item())

        print(f"epoch={epoch}  avg_loss={total/len(train_loader):.4f}")

    # -------- inference on valid: retrieval on frozen item_embs --------
    model.eval()
    Ks = [1, 10, 50, 100]
    preds_by_K = {K: [] for K in Ks}

    with torch.no_grad():
        for seq in valid_seqs:
            # use last up to 20 items as prefix
            prefix = [i for i in seq if i in id2idx][-20:]
            if len(prefix) == 0:
                # fallback: arbitrary
                for K in Ks:
                    preds_by_K[K].append(item_ids[:K].tolist())
                continue

            # build x_idx [1,T]
            x = torch.tensor([[id2idx[i] for i in prefix]], dtype=torch.long, device=device)
            lengths = torch.tensor([x.shape[1]], dtype=torch.long, device=device)
            x_emb = item_embs[x]

            u = model(x_emb, lengths).squeeze(0)        # [D]
            scores = (item_embs @ u).detach().cpu().numpy()  # [N]

            # exclude seen
            seen = set(seq)
            for it in seen:
                idx = id2idx.get(it)
                if idx is not None:
                    scores[idx] = -1e9

            for K in Ks:
                top_idx = np.argpartition(-scores, K)[:K]
                top_idx = top_idx[np.argsort(-scores[top_idx])]
                preds_by_K[K].append(item_ids[top_idx].tolist())

    for K in Ks:
        r, n = recall_ndcg_at_k(valid_tgts, preds_by_K[K], K)
        print(f"K={K:3d}  Recall@K={r:.4f}  NDCG@K={n:.4f}")

    # save_orbit_bin(preds_by_K[100], 100, "submissions/gru_valid_K100.bin")
    # save_orbit_bin(preds_by_K[10], 10, "submissions/gru_valid_K10.bin")

        # ----- Clustering / plots sur les embeddings de session GRU -----
    U = compute_gru_session_embeddings(
        model=model,
        valid_seqs=valid_seqs,
        id2idx=id2idx,
        item_ids=item_ids,
        item_embs=item_embs,
        device=device,
        max_prefix_len=20,
    )

    kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(U)

    plot_tsne_clusters(U, labels, out_path="gru_user_tsne_30.png", max_points=2000)
    summarize_clusters(labels, valid_tgts, preds_by_K[100], out_csv="gru_cluster_summary_30.csv")

    id2url = load_item_text_simple("../data/ClueWeb-Reco/item_text.tsv")

    print("id2url size =", len(id2url))
    print("example seq ids =", valid_seqs[0][:5])
    print("mapping hit example =", id2url.get(int(valid_seqs[0][0])) if valid_seqs and valid_seqs[0] else None)


    # export_cluster_websites(
    #     labels,
    #     valid_seqs, 
    #     id2url,
    #     out_csv="gru_cluster_domains_30.csv",
    # )

if __name__ == "__main__":
    main()
