# SASRecF Implementation Guide

This README explains the additions and modifications made in the TASTE branch of the forked RecSys-Benchmark repo for the custom SASRecF model within RecBole.

## How to train SasRecF 

To run the full training pipeline from the RecBole directory:

```bash
cd RecBole/
bash scripts/run_SASRecF.sh
```

This launches SLURM job on Amazon Books (You can can refer to run_SASRecF.sh to uncomment/comment dataset choice lines to choose a different dataset) using our SASRecF configs, verifies CUDA/data, and produces logs/metrics/checkpoints.

## Key Additions/Modifications

Our work focused on SASRecF enhancements:

| File/Directory | Location | Purpose |
|---------------|----------|---------|
| `run_SASRecF.sh` | `scripts/` | SLURM launcher with CUDA checks, data verification, PYTHONPATH export, calls run_recbole.py with triple configs |
| `SASRecF.yaml` | `configs/models/` | Model overrides: 4 layers/heads, hidden_size=256, inner_size=1024, dropouts=0.3 (attn/hidden), initializer_range=0.01, layer_norm_eps=1e-8, selected_features=['categories'], pooling_mode=mean, loss_type=CE, aap_weight=1.0, mip_weight=0.2, epochs=1000, stopping_step=60, batch_size=2048/4096, clip_grad_norm=1 |
| `amzn.yaml` | `configs/datasets/` | Amazon Books config: features/categories, MIN interactions required, data_path/load_col set (you can change the amzn data type to another one among beauty,toys or sports) | 
| `ml.yaml` | `configs/datasets/` | MovieLens dataset config: features include 'movie_title','release_year' and'genre', MIN interactions resuired, data_path/load_col set| 
| `SASRecF.yaml` | `recbole/properties/model/` | SASRecF property notes (baseline vs our hyperparams for comparaison) | 
| `run_recbole.py` | (root) | CLI runner with nargs="*" for --config_files (supports model+dataset+eval multi-config) | 
| `plot-interpret.py` | `scripts/` | Parses .err/.log for losses/scores/debug tensors (embeddings/gates), plots curves/gate evolution, exports CSV metrics | 
| `sasrecf_ml1m_train-28019_metrics.csv` | `scripts/outputs_sasrecf_on_ml-1m/` | Example full ML-1M run metrics CSV for reference curves |
| `outputs_sasrecf_on_ml-1m/` | `scripts/` | ML-1M runs : .err/.out, _metrics.csv, _training.png, _health.png | 

## Chronological Run Order

1.
2. **Data handling**: Set Amazon split in `configs/datasets/amzn.yaml` (data_path/load_col for Books; chnage domains to beauty, toys or sports if needed). For MovieLens dataset use `configs/datasets/ml.yaml`
3. **Hyperparams**: Load `SASRecF.yaml` for transformer depth/width, dropouts, fusion (categories/mean), loss/weights, extended training
4. **Launch**: `sbatch scripts/run_SASRecF.sh` (verifies files/CUDA, runs triple-config), you can uncomment specified lines to change the dataset
5. **Monitor**: Post-run, `python scripts/plot-interpret.py <logfile>` extracts/plots losses, valid scores, gates; exports CSV 
6. **Reference**: We provided example metrics per epoch for ML-1M dataset (`sasrecf_ml1m_train-28019_metrics.csv`) and curves in PNG format

