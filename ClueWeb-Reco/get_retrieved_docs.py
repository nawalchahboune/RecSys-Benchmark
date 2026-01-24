import requests
import base64
import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

URL = "https://clueweb22.us/search"
API_KEY = os.environ.get("CLUEWEB_API_KEY", "YOUR_API_KEY")
FOLDER = os.environ.get("CLUEWEB_INPUT_DIR", "/home/karrym/capstone/RecSys-Benchmark/ClueWeb-Reco/outputs_gpt_41")
MAP_FILE = os.environ.get("CLUEWEB_MAPPING_PATH", "/data/group_data/cx_group/REC/ClueWeb-Reco/ClueWeb-Reco_public/cwid_to_id.tsv")

def write_embed_to_binary(embeddings, output_path):
    num, dim = embeddings.shape
    with open(output_path, "wb") as f:
        f.write(num.to_bytes(4, "little"))
        f.write(dim.to_bytes(4, "little"))
        f.write(embeddings.astype(np.int32).tobytes())

def retrive(query, k=100):
    resp = requests.get(URL, params={"query": query, "k": k}, headers={"X-API-Key": API_KEY})
    resp.raise_for_status()
    data = resp.json()
    docs = []
    for item in data.get("results", []):
        doc = base64.b64decode(item).decode("utf-8")
        docs.append(json.loads(doc))
    return docs

def process_queries(input_file, output_file, k=100):
    if os.path.exists(output_file):
        df = pd.read_csv(output_file, sep="\t")
        # ajuster nb de colonnes si K a changé
        existing = len([c for c in df.columns if c.startswith("retrieved_")])
        for i in range(existing, k):
            df[f"retrieved_{i}"] = None
    else:
        df = pd.read_csv(input_file, header=None, names=["session_id", "query"], sep="\t")
        for i in range(k):
            df[f"retrieved_{i}"] = None

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {os.path.basename(input_file)}"):
        if pd.notna(row.get("retrieved_0")):
            continue
        query = row["query"]
        while True:
            try:
                retrieved_docs = retrive(query, k)
                if len(retrieved_docs) != k:
                    raise RuntimeError(f"Expected {k} docs, got {len(retrieved_docs)}")
                for i, doc in enumerate(retrieved_docs):
                    df.at[index, f"retrieved_{i}"] = doc["ClueWeb22-ID"].strip()
                break
            except Exception as e:
                print(f"[ERROR] Query failed at index {index}: {e}")
                continue
        if index % 10 == 0:
            df.to_csv(output_file, index=False, sep="\t")
    df.to_csv(output_file, index=False, sep="\t")

def load_needed_mapping(mapping_path, needed_tokens):
    tok2id = {}
    with open(mapping_path, "r") as f:
        for line in f:
            cwid, cid = line.rstrip("\n").split("\t")
            if cwid in needed_tokens:
                tok2id[cwid] = int(cid)
    return tok2id

def save_result(retrieved_df, output_path):
    if os.path.exists(output_path):
        print(f"Binary file already exists: {output_path}, skipping.")
        return
    df = pd.read_csv(retrieved_df, sep="\t")
    k_cols = [c for c in df.columns if c.startswith("retrieved_")]
    k = len(k_cols)
    # collecter tous les CWIDs nécessaires puis construire un mapping minimal
    needed = set()
    for col in k_cols:
        needed.update(map(lambda x: str(x).strip(), df[col].dropna().tolist()))
    tok2id = load_needed_mapping(MAP_FILE, needed)

    results = []
    for i in range(len(df)):
        row = []
        ok = True
        for col in k_cols:
            cwid = df.iloc[i][col]
            if pd.isna(cwid):
                ok = False; break
            row.append(tok2id.get(str(cwid).strip(), 0))  # 0 si manquant
        if ok:
            results.append(row)
    results = np.asarray(results, dtype=np.int32)
    print("results shape: ", results.shape)  # (num_sessions, k)
    write_embed_to_binary(results, output_path)

if __name__ == "__main__":
    K = int(os.environ.get("SUBMISSION_K", "100"))
    for file in os.listdir(FOLDER):
        if file.endswith(".tsv") and not file.startswith("retrieved_"):
            input_file = os.path.join(FOLDER, file)
            retrieved_file = os.path.join(FOLDER, f"retrieved_{file}")
            bin_file = retrieved_file.replace(".tsv", ".bin")
            print(f"\nProcessing {input_file}")
            process_queries(input_file, retrieved_file, k=K)
            save_result(retrieved_file, bin_file)