# SASRecF Implementation Guide

This README explains the additions and modifications made in the TASTE branch of the forked RecSys-Benchmark repo for the custom SASRecF model within RecBole framework.
N.B: all concerned files are within the RecBole directory.

## How to train SasRecF 

To run the full training pipeline from the RecBole directory:

```bash
cd RecBole/
bash scripts/run_SASRecF.sh
```

This launches SLURM job on Amazon Books using our SASRecF configs, verifies CUDA/data, and produces logs/metrics/checkpoints (You can can refer to run_SASRecF.sh to uncomment/comment dataset choice lines to choose a different dataset).

## How to run SasRecF inference 

If you want to quickly run inference on our trained model:

```bash
cd RecBole/quick_start/
python quick_start.py --eval_only True
```

The quick_start.py script has been configured with an eval_only flag that allows you to skip training and directly evaluate predictions. This is the fastest way to see model results(don't foget to change your custom dataset and checkpoint paths).

## Key Additions/Modifications

Our work focused on SASRecF enhancements:

| File/Directory | Location | Purpose |
|---------------|----------|---------|
| `run_SASRecF.sh` | `RecBole/scripts/` | SLURM launcher with CUDA checks, data verification, PYTHONPATH export, calls run_recbole.py with triple configs |
| `SASRecF.yaml` | `RecBole/configs/models/` | Model overrides: 4 layers/heads, hidden_size=256, inner_size=1024, dropouts=0.3 (attn/hidden), initializer_range=0.01, layer_norm_eps=1e-8, selected_features=['categories'], pooling_mode=mean, loss_type=CE, aap_weight=1.0, mip_weight=0.2, epochs=1000, stopping_step=60, batch_size=2048/4096, clip_grad_norm=1 |
| `amzn.yaml` | `RecBole/configs/datasets/` | Amazon Books config: features/categories, MIN interactions required, data_path/load_col set (you can change the amzn data type to another one among beauty,toys or sports) | 
| `ml.yaml` | `RecBole/configs/datasets/` | MovieLens dataset config: features include 'movie_title','release_year' and'genre', MIN interactions resuired, data_path/load_col set| 
| `SASRecF.yaml` | `RecBole/recbole/properties/model/` | SASRecF property notes (baseline vs our hyperparams for comparaison) | 
| `run_recbole.py` | root (RecBole) | CLI runner with nargs="*" for --config_files (supports model+dataset+eval multi-config) | 
| `plot-interpret.py` | `RecBole/scripts/` | Parses .err/.log for losses/scores/debug tensors (embeddings/gates), plots curves/gate evolution, exports CSV metrics | 
| `sasrecf_ml1m_train-28019_metrics.csv` | `RecBole/scripts/outputs_sasrecf_on_ml-1m/` | Example full ML-1M run metrics CSV for reference curves |
| `outputs_sasrecf_on_ml-1m/` | `RecBole/scripts/` | ML-1M runs : .err/.out, _metrics.csv, _training.png, _health.png | 

## Chronological Run Order

1. **Uploading the Data** Please upload the data to the root directory (RecSys-Benchmark) following instructions in the main README.md and make sure to respect the naming standard (e.g. `data-amzn_beauty/amzn/amzn.item` and `data-amzn_beauty/amzn/amzn.iter` for Amazon Beauty for example).
2. **Data handling**: Set Amazon split in `RecBole/configs/datasets/amzn.yaml` (data_path/load_col for Books; chnage domains to beauty, toys or sports if needed). For MovieLens dataset use `RecBole/configs/datasets/ml.yaml`
3. **Hyperparams**: Load `SASRecF.yaml` for transformer depth/width, dropouts, fusion (categories/mean), loss/weights, extended training
4. **Launch**: `sbatch RecBole/scripts/run_SASRecF.sh` (verifies files/CUDA, runs triple-config), you can uncomment specified lines to change the dataset
5. **Monitor**: Post-run, `python RecBole/scripts/plot-interpret.py <logfile>` extracts/plots losses, valid scores, gates; exports CSV 
6. **Reference**: We provided example metrics per epoch for ML-1M dataset (`sasrecf_ml1m_train-28019_metrics.csv`) and curves in PNG format

## Troubleshooting: RecBole changes not applied

If you modify RecBole source code (e.g. a custom model or Config) but the changes are not used at runtime, Python is likely importing RecBole from site-packages instead of your local source tree.

To force the runtime to use your modified files, copy them into the active virtual environment:
```bash
cp RecBole/recbole/model/sequential_recommender/sasrecf.py \
   $VIRTUAL_ENV/lib/python3.10/site-packages/RecBole/recbole/model/sequential_recommender/sasrecf.py
```
```bash
cp RecBole/recbole/config/configurator.py \
   $VIRTUAL_ENV/lib/python3.10/site-packages/RecBole/recbole/config/configurator.py
```
