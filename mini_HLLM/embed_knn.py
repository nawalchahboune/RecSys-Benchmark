from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def read_valid_ordered(input_path: Path, target_path: Path) -> tuple[list[list[int]], list[int]]:
    def parse_hist(s: str) -> list[int]:
        s = s.strip()
        if not s:
            return []
        if "," in s:
            return [int(x) for x in s.split(",") if x.strip()]
        return [int(x) for x in s.split() if x.strip()]

    seqs: list[list[int]] = []
    tgts: list[int] = []

    with input_path.open("r", encoding="utf-8") as fin, target_path.open("r", encoding="utf-8") as ftgt:
        h_in = fin.readline()
        h_tgt = ftgt.readline()
        for line_in, line_tgt in zip(fin, ftgt):
            parts_in = line_in.strip().split("\t")
            parts_tgt = line_tgt.strip().split("\t")
            if len(parts_in) < 2 or len(parts_tgt) < 2:
                continue
            seqs.append(parse_hist(parts_in[1]))
            tgts.append(int(parts_tgt[1]))
    assert len(seqs) == len(tgts)
    return seqs, tgts


def build_id2idx(item_ids: np.ndarray) -> Dict[int, int]:
    return {int(i): idx for idx, i in enumerate(item_ids)}


def session_user_emb(seq: list[int], id2idx: dict[int, int], item_embs: np.ndarray, max_len: int = 10) -> np.ndarray | None:
    seq = [i for i in seq if i in id2idx]
    if not seq:
        return None
    seq = seq[-max_len:]
    embs = item_embs[[id2idx[i] for i in seq]]  # (L, D)
    u = embs.mean(axis=0)
    norm = np.linalg.norm(u)
    if norm > 0:
        u = u / norm
    return u


def recommend_topk_for_seq(
    seq: list[int],
    item_ids: np.ndarray,
    item_embs: np.ndarray,
    id2idx: dict[int, int],
    K: int,
    max_len: int = 10,
    exclude_seen: bool = True,
) -> list[int]:
    u = session_user_emb(seq, id2idx, item_embs, max_len=max_len)
    if u is None:
        base = item_ids[:K].astype(np.int64).tolist()
        if len(base) < K:
            base += [0] * (K - len(base))
        return base

    scores = item_embs @ u  # (N,)
    if exclude_seen:
        seen = set(seq)
        for it in seen:
            idx = id2idx.get(it)
            if idx is not None:
                scores[idx] = -1e9

    if K >= scores.shape[0]:
        top_idx = np.argsort(-scores)
    else:
        top_idx = np.argpartition(-scores, K)[:K]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

    return item_ids[top_idx].astype(np.int64).tolist()


def recall_at_k(targets: list[int], preds: list[list[int]], K: int) -> float:
    hits = 0
    n = len(targets)
    for t, row in zip(targets, preds):
        if t in row[:K]:
            hits += 1
    return hits / n if n > 0 else 0.0


def ndcg_at_k(targets: list[int], preds: list[list[int]], K: int) -> float:
    import math
    n = len(targets)
    total = 0.0
    for t, row in zip(targets, preds):
        score = 0.0
        for rank, item in enumerate(row[:K], start=1):
            if item == t:
                score = 1.0 / math.log2(rank + 1)
                break
        # IDCG (max possible) = 1/log2(1+1) = 1
        total += score
    return total / n if n > 0 else 0.0


def compute_user_embeddings(
    seqs: list[list[int]],
    id2idx: dict[int, int],
    item_embs: np.ndarray,
    max_len: int = 10,
) -> np.ndarray:
    """
    Retourne un array [N_sessions, D] de user embeddings (moyenne des items vus).
    Si une session ne contient aucun item connu, on la met à 0.
    """
    D = item_embs.shape[1]
    users = []
    for seq in seqs:
        u = session_user_emb(seq, id2idx, item_embs, max_len=max_len)
        if u is None:
            u = np.zeros(D, dtype=np.float32)
        users.append(u.astype(np.float32))
    return np.stack(users, axis=0)  # [N, D]

def cluster_and_plot(user_embs: np.ndarray, n_clusters: int = 10, max_points: int = 2000):
    """
    - KMeans pour clusteriser les user embeddings.
    - t-SNE sur 2D pour visualiser.
    - Scatter où la couleur = cluster.
    """
    N, D = user_embs.shape
    print("User emb shape:", user_embs.shape)

    idx = np.arange(N)
    if N > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=max_points, replace=False)
    sub_embs = user_embs[idx]

    # KMeans sur tous les users 
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    labels_all = kmeans.fit_predict(user_embs)
    labels = labels_all[idx]

    # t-SNE sur subset
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0, init="pca")
    coords = tsne.fit_transform(sub_embs)  # [M, 2]

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=10, cmap="tab10", alpha=0.7)
    plt.colorbar(label="Cluster ID")
    plt.title("User embedding clusters (t-SNE 2D)")
    plt.tight_layout()
    plt.savefig("user_clusters_tsne_30.png", dpi=200)
    plt.close()
    print("Saved plot: user_clusters_tsne_30.png")


def main():
    base = Path("../data/ClueWeb-Reco")
    ordered_dir = base / "ordered_id_splits"

    valid_input = ordered_dir / "valid_input.tsv"
    valid_target = ordered_dir / "valid_target.tsv"

    seqs, tgts = read_valid_ordered(valid_input, valid_target)
    print("Valid examples:", len(seqs))

    artifacts = Path("artifacts")
    item_ids = np.load(artifacts / "item_ids_30.npy")
    item_embs = np.load(artifacts / "item_embs_30.npy")

    id2idx = build_id2idx(item_ids)
    print("Item table:", len(item_ids), "emb dim:", item_embs.shape[1])

    # 3) métriques 
    Ks = [1, 10, 50, 100]
    preds_by_K: dict[int, list[list[int]]] = {K: [] for K in Ks}

    for seq in seqs:
        for K in Ks:
            row = recommend_topk_for_seq(
                seq=seq,
                item_ids=item_ids,
                item_embs=item_embs,
                id2idx=id2idx,
                K=K,
                max_len=10,
                exclude_seen=True,
            )
            preds_by_K[K].append(row)

    for K in Ks:
        rec = recall_at_k(tgts, preds_by_K[K], K)
        ndcg = ndcg_at_k(tgts, preds_by_K[K], K)
        print(f"K={K:3d}  Recall@K={rec:.4f}  NDCG@K={ndcg:.4f}")

    # 5) user embeddings + clustering + plots
    user_embs = compute_user_embeddings(seqs, id2idx, item_embs, max_len=10)
    cluster_and_plot(user_embs, n_clusters=10, max_points=2000)


if __name__ == "__main__":
    main()
