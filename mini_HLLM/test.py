from pathlib import Path

def load_valid_targets(path):
    tgts = []
    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                tgts.append(int(parts[1]))
    return tgts

import numpy as np

item_ids = set(np.load("artifacts/item_ids.npy").astype(int).tolist())
tgts = load_valid_targets("../data/ClueWeb-Reco/ordered_id_splits/valid_target.tsv")
coverage = sum(t in item_ids for t in tgts) / len(tgts)
print("valid target coverage:", coverage)
