import struct
from collections import defaultdict, Counter
import math

def parse_hist(s: str):
    s = s.strip()
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    return [int(x) for x in s.split() if x.strip()]

def read_ordered_tsv(path):
    sessions = []
    with open(path, "r", encoding="utf-8") as f:
        _ = f.readline()  # header
        for line in f:
            sid, hist = line.rstrip("\n").split("\t")
            sessions.append(parse_hist(hist))
    return sessions

def write_bin(out_path, all_preds, K):
    n = len(all_preds)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(struct.pack("<I", K))
        for row in all_preds:
            row = list(row)
            if len(row) < K:
                pad = row[-1] if len(row) > 0 else 0
                row = row + [pad] * (K - len(row))
            else:
                row = row[:K]
            for x in row:
                f.write(struct.pack("<i", int(x)))


def build_item_to_sessions(train_sessions):
    item_to_sids = defaultdict(set)
    for sid, seq in enumerate(train_sessions):
        for it in set(seq):  # set -> éviter compter 2 fois dans même session
            item_to_sids[it].add(sid)
    return item_to_sids

def sknn_predict(train_sessions, test_sessions, K=100, m=10, topN=2000):
    """
    m = combien de derniers items on utilise pour trouver des voisins
    topN = limite du nombre de sessions candidates pour rester rapide
    """
    item_to_sids = build_item_to_sessions(train_sessions)

    # fallback global popularity (au cas où)
    pop = Counter()
    for seq in train_sessions:
        pop.update(seq)
    global_pop = [it for it, _ in pop.most_common(K)]

    preds = []
    for seq in test_sessions:
        if not seq:
            preds.append(global_pop[:K])
            continue

        recent = seq[-m:] if len(seq) >= m else seq
        recent_set = set(recent)

        # 1) candidats = union des sessions qui contiennent ces items récents
        cand = set()
        for it in recent_set:
            cand |= item_to_sids.get(it, set())

        # si trop de candidats, on garde ceux qui matchent le plus (approx rapide)
        if len(cand) > topN:
            # score rapide = nb d'items récents présents dans la session
            tmp = []
            for sid in cand:
                inter = recent_set.intersection(train_sessions[sid])
                tmp.append((len(inter), sid))
            tmp.sort(reverse=True)
            cand = {sid for _, sid in tmp[:topN]}

        # 2) score voisin = cosine sur sets (ici: intersection / sqrt(|A||B|))
        neigh_scores = []
        for sid in cand:
            s = train_sessions[sid]
            sset = set(s)
            inter = len(recent_set & sset)
            if inter == 0:
                continue
            score = inter / math.sqrt(len(recent_set) * len(sset))
            neigh_scores.append((score, sid))
        neigh_scores.sort(reverse=True)

        # 3) vote items à partir des voisins
        item_scores = defaultdict(float)
        for score, sid in neigh_scores[:200]:  # prends top voisins
            for it in train_sessions[sid]:
                if it in recent_set:  # option: ne pas recommander déjà vu
                    continue
                item_scores[it] += score

        ranked = [it for it, _ in sorted(item_scores.items(), key=lambda x: x[1], reverse=True)]

        # fallback si pas assez
        if len(ranked) < K:
            for it in global_pop:
                if it not in recent_set and it not in ranked:
                    ranked.append(it)
                if len(ranked) == K:
                    break

        preds.append(ranked[:K])
    return preds

if __name__ == "__main__":
    # Pour ClueWeb-Reco: on a surtout valid_input/test_input dans ordered_id_splits
    valid_path = "ordered_id_splits/valid_input.tsv"
    test_path  = "ordered_id_splits/test_input.tsv"

    # "train" n'existe pas toujours pour ClueWeb-Reco.
    # Donc on utilise le valid_input comme "base" (proxy train) :
    train_sessions = read_ordered_tsv(valid_path)

    # 1) soumission validation (auto-évaluable via le site)
    valid_sessions = read_ordered_tsv(valid_path)
    for K in [10, 50, 100]:
        preds = sknn_predict(train_sessions, valid_sessions, K=K, m=10, topN=2000)
        out = f"sknn_valid_K{K}.bin"
        write_bin(out, preds, K)
        print("✅ wrote", out, "n=", len(preds), "K=", K)

    # 2) soumission test (score caché)
    try:
        test_sessions = read_ordered_tsv(test_path)
        for K in [10, 50, 100]:
            preds = sknn_predict(train_sessions, test_sessions, K=K, m=10, topN=2000)
            out = f"sknn_test_K{K}.bin"
            write_bin(out, preds, K)
            print("✅ wrote", out, "n=", len(preds), "K=", K)
    except FileNotFoundError:
        print("ℹ️ test_input.tsv not found, only wrote validation bins.")