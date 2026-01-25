from __future__ import annotations

import math
import csv
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def read_interactions(path: Path) -> Dict[str, List[int]]:
    """
    Expects TSV with header:
      session_id \t cw_internal_id \t timestamp
    Returns dict: session_id -> list of item_ids (assumed ordered already).
    """
    sess = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        # header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sid = parts[0]
            iid = int(parts[1])
            sess[sid].append(iid)
    return sess


def read_ordered_tsv(path: Path) -> List[List[int]]:
    """
    ordered_id_splits/valid_input.tsv:
      session_id \t ordered_history_cw_internal_id
    history can be "1 2 3" or "1,2,3"
    Returns list of sequences in file order.
    """
    def parse_hist(s: str) -> List[int]:
        s = s.strip()
        if not s:
            return []
        if "," in s:
            return [int(x) for x in s.split(",") if x.strip()]
        return [int(x) for x in s.split() if x.strip()]

    sessions: List[List[int]] = []
    with path.open("r", encoding="utf-8") as f:
        # header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sessions.append(parse_hist(parts[1]) if len(parts) > 1 else [])
    return sessions


def read_targets(path: Path) -> List[int]:
    """
    ordered_id_splits/valid_target.tsv:
      session_id \t target_cw_internal_id
    """
    t: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        # header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                t.append(int(parts[1]))
    return t


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)


def session_emb_mean(seq: List[int], id2idx: Dict[int, int], item_embs: np.ndarray) -> np.ndarray:
    idx = [id2idx[i] for i in seq if i in id2idx]
    if not idx:
        return np.zeros((item_embs.shape[1],), dtype=np.float32)
    return item_embs[idx].mean(axis=0)


def fit_linear_markov(X: np.ndarray, Y: np.ndarray, lam: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit Y ≈ X W + b  (multi-output ridge regression).
    Returns W [d,d], b [d]
    """
    N, d = X.shape
    X_aug = np.concatenate([X, np.ones((N, 1), dtype=X.dtype)], axis=1)  # [N, d+1]

    A = X_aug.T @ X_aug + lam * np.eye(d + 1, dtype=X.dtype)
    B = X_aug.T @ Y
    W_full = np.linalg.solve(A, B)  # [d+1, d]

    W = W_full[:-1, :]  # [d, d]
    b = W_full[-1, :]   # [d]
    return W, b


def build_transition_dataset(train_sessions: Dict[str, List[int]], id2idx: Dict[int, int], item_embs: np.ndarray):
    X_list, Y_list = [], []
    for seq in train_sessions.values():
        for a, b in zip(seq[:-1], seq[1:]):
            if a in id2idx and b in id2idx:
                X_list.append(item_embs[id2idx[a]])
                Y_list.append(item_embs[id2idx[b]])
    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = np.stack(Y_list, axis=0).astype(np.float32)
    return X, Y


def predict_next_emb(last_item_id: int, id2idx: Dict[int, int], item_embs: np.ndarray, W: np.ndarray, b: np.ndarray):
    if last_item_id not in id2idx:
        return None
    x = item_embs[id2idx[last_item_id]]
    return x @ W + b  # [d]


def topk_cosine(query: np.ndarray, item_embs_norm: np.ndarray, K: int) -> np.ndarray:
    """
    query: [d] assumed L2-normalized 
    item_embs_norm: [n_items, d] L2-normalized
    Returns indices of top K by dot product.
    """
    q = query / (np.linalg.norm(query) + 1e-12)
    scores = item_embs_norm @ q
    idx = np.argpartition(-scores, K-1)[:K]
    idx = idx[np.argsort(-scores[idx])]
    return idx


def recommend_cont_markov(
    seq: List[int],
    item_ids: np.ndarray,
    item_embs_norm: np.ndarray,
    id2idx: Dict[int, int],
    W: np.ndarray,
    b: np.ndarray,
    K: int,
    exclude_seen: bool = True,
) -> List[int]:
    used = set(seq) if exclude_seen else set()
    if not seq:
        return [int(x) for x in item_ids[:K]]

    e_pred = predict_next_emb(seq[-1], id2idx, item_embs_norm, W, b)
    if e_pred is None:
        return [int(x) for x in item_ids[:K]]

    top_idx = topk_cosine(e_pred, item_embs_norm, K=min(K*5, len(item_ids)))  # shortlist
    out = []
    for ix in top_idx:
        iid = int(item_ids[ix])
        if iid not in used:
            out.append(iid)
        if len(out) == K:
            break

    if len(out) < K:
        out += [0] * (K - len(out))
    return out


def recall_at_k(targets: List[int], preds: List[List[int]], K: int) -> float:
    if not targets:
        return 0.0
    return sum(1 for t, row in zip(targets, preds) if t in row[:K]) / len(targets)


def ndcg_at_k(targets: List[int], preds: List[List[int]], K: int) -> float:
    if not targets:
        return 0.0
    total = 0.0
    for t, row in zip(targets, preds):
        topk = row[:K]
        if t in topk:
            r = topk.index(t) + 1
            total += 1.0 / math.log2(r + 1)
    return total / len(targets)


def plot_tsne_clusters(U: np.ndarray, labels: np.ndarray, out_path: str, max_points: int = 3000):
    N = U.shape[0]
    idx = np.arange(N)
    if N > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=max_points, replace=False)

    U_sub = U[idx]
    lab_sub = labels[idx]

    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=0)
    Z = tsne.fit_transform(U_sub)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=lab_sub, s=10, alpha=0.7, cmap="tab10")
    plt.colorbar(sc, label="Cluster")
    plt.title("Session embeddings clustered (t-SNE)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)


def cluster_summary_csv(labels: np.ndarray, targets: List[int], preds_at_100: List[List[int]], out_csv: str):
    by_c = defaultdict(list)
    for i, c in enumerate(labels):
        by_c[int(c)].append(i)

    rows = []
    for c, idxs in sorted(by_c.items()):
        n = len(idxs)
        hit100 = sum(1 for i in idxs if targets[i] in preds_at_100[i]) / n if n else 0.0
        rows.append({"cluster": c, "n_sessions": n, "hit@100": hit100})

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cluster", "n_sessions", "hit@100"])
        w.writeheader()
        w.writerows(rows)
    print("Saved:", out_csv)


def main():
    base = Path("../data/ClueWeb-Reco")
    inter_dir = base / "interaction_splits"
    ordered_dir = base / "ordered_id_splits"
    artifacts = Path("artifacts")


    train_path = inter_dir / "valid_inter_input.tsv"
    valid_input = ordered_dir / "valid_input.tsv"
    valid_target = ordered_dir / "valid_target.tsv"

    item_ids_path = artifacts / "item_ids_30.npy"
    item_embs_path = artifacts / "item_embs_30.npy"

    item_ids = np.load(item_ids_path)  # [n_items]
    item_embs = np.load(item_embs_path).astype(np.float32)  # [n_items, d]
    item_embs_norm = l2_normalize_rows(item_embs)

    id2idx = {int(item_ids[i]): i for i in range(len(item_ids))}

    train_sessions = read_interactions(train_path)
    X, Y = build_transition_dataset(train_sessions, id2idx, item_embs_norm)
    W, b = fit_linear_markov(X, Y, lam=1e-2)  # you can tune lam
    print("Transitions used:", X.shape[0], "dim:", X.shape[1])

    seqs = read_ordered_tsv(valid_input)
    tgts = read_targets(valid_target)

    Ks = [1, 10, 50, 100]
    preds_by_K = {}
    for K in Ks:
        preds = [
            recommend_cont_markov(seq, item_ids, item_embs_norm, id2idx, W, b, K=K, exclude_seen=True)
            for seq in seqs
        ]
        preds_by_K[K] = preds
        print(f"K={K:3d}  Recall@K={recall_at_k(tgts, preds, K):.4f}  NDCG@K={ndcg_at_k(tgts, preds, K):.4f}")

    U = np.stack([session_emb_mean(seq, id2idx, item_embs_norm) for seq in seqs], axis=0)
    U = l2_normalize_rows(U)

    kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto")  
    labels = kmeans.fit_predict(U)

    plot_tsne_clusters(U, labels, out_path="cont_markov_session_tsne_30.png", max_points=3000)
    cluster_summary_csv(labels, tgts, preds_by_K[100], out_csv="cont_markov_cluster_summary_30.csv")


if __name__ == "__main__":
    main()
