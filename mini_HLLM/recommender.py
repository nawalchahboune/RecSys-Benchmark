import numpy as np

item_ids = np.load("artifacts/item_ids.npy")          
item_embs = np.load("artifacts/item_embs.npy")       

# mapping id -> index
id2idx = {int(i): idx for idx, i in enumerate(item_ids)}

def sequence_to_user_emb(seq, max_len=10):
    # garder les derniers max_len items
    seq = [i for i in seq if i in id2idx]
    if not seq:
        return None
    seq = seq[-max_len:]

    embs = np.stack([item_embs[id2idx[i]] for i in seq], axis=0)  # [L, D]
    # moyenne
    user_emb = embs.mean(axis=0)                                  # [D]
    # renormaliser (optionnel mais utile pour cosinus)
    norm = np.linalg.norm(user_emb)
    if norm > 0:
        user_emb = user_emb / norm
    return user_emb


def recommend_topk_for_seq(seq, k=50):
    user_emb = sequence_to_user_emb(seq)
    if user_emb is None:
        return []

    # scores = produit scalaire user_emb · item_embs^T
    scores = item_embs @ user_emb  # [N_items]

    # exclure les items déjà vus
    seen_set = set(seq)
    for item in seen_set:
        idx = id2idx.get(item)
        if idx is not None:
            scores[idx] = -1e9

    # top-K indexes
    if k >= len(scores):
        top_idx = np.argsort(-scores)
    else:
        top_idx = np.argpartition(-scores, k)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

    top_items = item_ids[top_idx]
    top_scores = scores[top_idx]
    return list(zip(top_items, top_scores))


