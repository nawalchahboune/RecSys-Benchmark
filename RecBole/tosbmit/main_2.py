import struct
from collections import defaultdict, Counter

# ---------- IO ----------
def read_interactions(path):
    """
    Expects TSV with header and columns like:
    session_id \t cw_internal_id \t timestamp
    Returns dict: session_id -> ordered list of item_ids (sorted by timestamp if needed).
    """
    sess = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sid = parts[0]
            iid = int(parts[1])
            # timestamp = parts[2]  # not used if file already ordered
            sess[sid].append(iid)
    return sess

def read_ordered_tsv(path):
    """
    ordered_id_splits/valid_input.tsv:
    session_id \t ordered_history_cw_internal_id
    history can be "1 2 3" or "1,2,3"
    Returns list of sequences in file order.
    """
    def parse_hist(s: str):
        s = s.strip()
        if not s:
            return []
        if "," in s:
            return [int(x) for x in s.split(",") if x.strip()]
        return [int(x) for x in s.split() if x.strip()]

    sessions = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            sid, hist = line.rstrip("\n").split("\t")
            sessions.append(parse_hist(hist))
    return sessions

def write_bin(out_path, all_preds, K):
    n = len(all_preds)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", n))  # num_sessions
        f.write(struct.pack("<I", K))  # K
        for row in all_preds:
            assert len(row) == K
            for x in row:
                f.write(struct.pack("<i", int(x)))  # int32

# ---------- Model: Markov + Popularity ----------
def train_markov_and_pop(train_sessions):
    """
    train_sessions: dict sid -> list of item_ids
    Returns:
      next_counts: dict last -> Counter(next_item)
      pop: Counter(item)
    """
    next_counts = defaultdict(Counter)
    pop = Counter()

    for sid, seq in train_sessions.items():
        for x in seq:
            pop[x] += 1
        for a, b in zip(seq[:-1], seq[1:]):
            next_counts[a][b] += 1

    return next_counts, pop

def predict_markov(seqs, next_counts, pop_list, K):
    preds = []
    pop_fallback = pop_list  # list sorted by popularity desc

    for seq in seqs:
        used = set(seq)  # optional: avoid recommending already seen
        if len(seq) == 0:
            row = []
        else:
            last = seq[-1]
            row = []
            # 1) Markov candidates
            if last in next_counts:
                for item, _c in next_counts[last].most_common():
                    if item not in used:
                        row.append(item)
                    if len(row) == K:
                        break

        # 2) Popularity fallback
        if len(row) < K:
            for item in pop_fallback:
                if item not in used:
                    row.append(item)
                if len(row) == K:
                    break

        # 3) Safety: still not enough? pad with 0
        if len(row) < K:
            row += [0] * (K - len(row))

        preds.append(row)

    return preds

if __name__ == "__main__":
    # 1) Train on train interactions
    train_path = "interaction_splits/valid_inter_input.tsv"  
    train_sessions = read_interactions(train_path)

    next_counts, pop = train_markov_and_pop(train_sessions)
    pop_list = [i for i, _ in pop.most_common()]

    # 2) Predict on validation input (ordered format)
    valid_seqs = read_ordered_tsv("ordered_id_splits/valid_input.tsv")

    for K in [10, 50, 100]:
        pred = predict_markov(valid_seqs, next_counts, pop_list, K)
        out = f"markov_pop_valid_K{K}.bin"
        write_bin(out, pred, K)
        print("✅ wrote", out, "n=", len(pred), "K=", K)