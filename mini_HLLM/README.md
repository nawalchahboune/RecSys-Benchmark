# mini_HLLM

A lightweight recommendation system implementation for the ClueWeb-Reco dataset, featuring multiple embedding-based and sequential approaches for next-item prediction in web browsing sessions.

## Results Summary

**Best performing model**: KNN with 30-epoch fine-tuned embeddings achieved the highest validation performance.

After experimenting with multiple approaches (KNN, GRU, Markov+KNN hybrid), the simple KNN-based method using embeddings from 30 epochs of fine-tuning significantly outperformed other configurations. The Markov+KNN hybrid and KNN with fewer training epochs (1 epoch) showed similar poor results and are not recommended for this dataset.

*Note: Further experimentation with additional training epochs was limited by available computational resources.*

## Overview

mini_HLLM provides several recommendation algorithms that leverage item embeddings and sequential patterns to predict the next webpage a user will visit. The project supports various approaches from simple KNN-based methods to more sophisticated GRU and Markov chain models.

## Features

- **Multiple Recommendation Approaches**:
  - KNN-based collaborative filtering with item embeddings
  - GRU (Gated Recurrent Unit) neural sequential modeling
  - Markov chain transition models
  - Combined embedding + Markov approaches

- **Item Embedding Pipeline**:
  - Fine-tuning sentence transformers on ClueWeb domain text
  - Generating normalized item embeddings
  - Support for various embedding dimensions

- **Flexible Data Processing**:
  - Handles multiple data formats (TSV, space-separated)
  - Automatic data downloading from HuggingFace
  - Sequence truncation and padding

## Project Structure

```
mini_HLLM/
├── data_script.py              # Download ClueWeb-Reco dataset from HuggingFace
├── item_encoder.py             # ClueWebSeqDataset class for loading sequences
├── helpers.py                  # Utility functions for data loading and preprocessing
├── main.py                     # Example usage of data loaders
├── test.py                     # Coverage testing utilities
│
├── finetune_item_embedder.py  # Fine-tune sentence transformer on triplet loss
├── item_embeds.py             # Generate item embeddings from fine-tuned model
│
├── embed_knn.py               # KNN-based recommendation using item embeddings
├── embed_GRU.py               # GRU sequential model with embeddings
├── embed_markov_KNN.py        # Combined Markov + KNN approach
│
├── all_seen_websites.py       # Extract all seen item IDs from splits
│
├── artifacts/                 # Generated embeddings and item IDs
├── checkpoints/               # Saved model checkpoints
├── finetuned_item_embedder/   # Fine-tuned sentence transformer models
├── pred_bins/                 # Binary prediction output files
└── submissions/               # Competition submission files
```

## Installation

1. **Clone the repository**:
```bash
cd RecSys-Benchmark/mini_HLLM
```

2. **Install dependencies**:
```bash
pip install torch numpy sentence-transformers scikit-learn matplotlib huggingface-hub
```

3. **Download the ClueWeb-Reco dataset**:
```bash
python data_script.py
```

This downloads the dataset to `../data/ClueWeb-Reco/` including:
- `ordered_id_splits/` - Train/valid/test splits with ordered sequences
- `interaction_splits/` - Raw interaction data
- `cwid_to_id.tsv` - Mapping from ClueWeb IDs to internal item IDs

## Quick Start

### 1. Generate Item Embeddings

First, fine-tune a sentence transformer on the ClueWeb domain data:

```bash
python finetune_item_embedder.py
```

This creates triplet training examples where:
- **Anchor**: Context of previous items in a session
- **Positive**: The actual next item visited
- **Negative**: A random item

Then generate embeddings for all seen items:

```bash
python item_embeds.py
```

Outputs saved to `artifacts/`:
- `item_ids_30.npy` - Array of item IDs
- `item_embs_30.npy` - Corresponding L2-normalized embeddings

### 2. Run KNN-based Recommendations

```bash
python embed_knn.py
```

This approach:
1. Loads pre-computed item embeddings
2. Represents each session as the mean of its item embeddings
3. Recommends items with highest cosine similarity to the session embedding
4. Evaluates Recall@K and NDCG@K metrics

### 3. Train GRU Sequential Model

```bash
python embed_GRU.py
```

Features:
- Uses pre-trained item embeddings as input
- GRU architecture learns sequential patterns
- Generates predictions for validation/test sets
- Includes visualization of learned user embeddings via t-SNE
- Performs domain clustering analysis

### 4. Markov + KNN Hybrid Model

```bash
python embed_markov_KNN.py
```

Combines:
- **Linear Markov transition model**: Predicts next embedding from current
- **KNN retrieval**: Finds nearest items to predicted embedding
- Handles continuous embedding space transitions

## Data Format

### Input Files (ordered_id_splits/)

**valid_input.tsv** / **train_input.tsv**:
```
session_id    ordered_history_cw_internal_id
sess_001      1,2,3,4,5
sess_002      10,20,30
```

**valid_target.tsv** / **train_target.tsv**:
```
session_id    target_cw_internal_id
sess_001      6
sess_002      40
```

**seen_item_ids.txt**:
```
1
2
3
...
```

### Output Format

Binary prediction files (`.bin`) use the ORBIT format:
```
<4 bytes: num_sessions>
<4 bytes: K>
<num_sessions * K * 4 bytes: predicted item IDs as int32>
```

## Key Components

### ClueWebSeqDataset

The core dataset class in [item_encoder.py](item_encoder.py) handles:
- Multiple TSV formats (with/without session IDs)
- Space and comma-separated sequences
- Sequence truncation to `max_seq_len`
- Left-padding for shorter sequences
- Optional target loading for train/valid splits

## Model Approaches

### 1. Embedding-based KNN
- **Idea**: Sessions with similar browsing history visit similar next pages
- **Method**: Average item embeddings → find nearest neighbor items
- **Pros**: Simple, interpretable, no training needed (after embeddings)
- **Cons**: Doesn't capture sequential order well

### 2. GRU Sequential Model
- **Idea**: Learn temporal patterns in browsing sequences
- **Architecture**: Embedding layer → GRU → Linear projection
- **Training**: Next-item prediction with cross-entropy loss
- **Pros**: Captures sequence dynamics, learns complex patterns
- **Cons**: Requires training, more computationally intensive

### 3. Markov + Embedding Hybrid
- **Idea**: Model transitions in continuous embedding space
- **Method**: Learn linear transformation W: current_emb → next_emb
- **Prediction**: Apply W, then KNN in embedding space
- **Pros**: Combines transition modeling with semantic similarity
- **Cons**: Linear assumption may be limiting

## Metrics

The project evaluates using standard ranking metrics:

- **Recall@K**: Fraction of sessions where target is in top-K predictions
- **NDCG@K**: Normalized Discounted Cumulative Gain at K

## Visualization

Several scripts generate t-SNE visualizations:
- User/session embeddings colored by domain clusters
- Item embedding space structure
- GRU hidden state analysis

Outputs:
- `user_clusters_tsne.png`
- `gru_user_tsne.png`
- `cont_markov_session_tsne.png`

## Configuration

Key hyperparameters to tune:

### Embedding Model
- `max_seq_length`: Max tokens for sentence transformer (default: 128)
- `embedding_dim`: Usually 384 for MiniLM, 768 for BERT-base
- Fine-tuning epochs: 30 (in `finetune_item_embedder.py`)

### GRU Model
- `hidden_dim`: GRU hidden size (default: 128)
- `n_layers`: Number of GRU layers (default: 2)
- `dropout`: Dropout rate (default: 0.3)
- `max_prefix_len`: Max sequence length for training pairs

### KNN
- `max_len`: Number of recent items to consider (default: 10)
- `exclude_seen`: Whether to exclude already-seen items (default: True)

## Troubleshooting

### Memory Issues
- Reduce `batch_size` in training/inference
- Use `max_examples` parameter in fine-tuning
- Process items in chunks

### Low Recall
- Increase `K` for evaluation
- Check item coverage in embeddings
- Tune `max_len` for session context