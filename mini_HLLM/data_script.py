from huggingface_hub import snapshot_download

REPO_ID = "cx-cmu/ClueWeb-Reco"

local_dir = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir="../data/ClueWeb-Reco",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "ordered_id_splits/*",
        "interaction_splits/*",
        "cwid_to_id.tsv",
        "cw_data_processing/*",
        "README*",
    ],
)

print("Downloaded to:", local_dir)