import os
import struct
import numpy as np
import torch

from recbole.quick_start import load_data_and_model

########################################
# CONFIG
########################################

MODEL_PTH_PATH = "checkpoints/SASRecF_ml-1m/SASRecF-Jan-17-2026_14-13-14.pth"
OUTPUT_BIN = "submission_ml1m.bin"
K = 10   # top-K demandé par ORBIT

########################################
# FORCE CPU (CRITIQUE)
########################################

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # empêche CUDA
os.environ["RECBOLE_USE_GPU"] = "False"

########################################
# LOAD MODEL + DATA (CPU)
########################################

print("[INFO] Loading trained model on CPU...")

checkpoint = torch.load(MODEL_PTH_PATH, map_location="cpu")

config = checkpoint["config"]

# 🔴 forçage dur côté RecBole
config["use_gpu"] = False
config["gpu_id"] = ""
config["device"] = torch.device("cpu")

# recharge via API officielle
from recbole.utils import init_seed
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model

init_seed(config["seed"], config["reproducibility"])

dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

model = get_model(config["model"])(config, train_data._dataset)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

########################################
# GENERATE TOP-K PREDICTIONS
########################################

print("[INFO] Generating predictions...")

preds = []

with torch.no_grad():
    for batch in test_data:
        interaction = batch[0] if isinstance(batch, (list, tuple)) else batch
        scores = model.full_sort_predict(interaction)
        topk = torch.topk(scores, K, dim=1).indices.cpu().numpy()
        preds.append(topk)

topk_matrix = np.vstack(preds)
num_sessions = topk_matrix.shape[0]

########################################
# ITEM ID → INTERNAL ID (ml-1m)
########################################

ds = train_data._dataset

# RecBole internal item ids sont déjà 0-index
internal_ids = topk_matrix.astype(np.int32)

########################################
# WRITE .bin (FORMAT ORBIT)
########################################

with open(OUTPUT_BIN, "wb") as f:
    f.write(struct.pack("<i", num_sessions))
    f.write(struct.pack("<i", K))
    f.write(internal_ids.ravel().tobytes())

print(f"[OK] submission saved: {OUTPUT_BIN}")
print(f"[INFO] sessions = {num_sessions}, K = {K}")
print(f"[INFO] expected size = {8 + num_sessions*K*4} bytes")
