# Latent Reasoning Directions (LRD)
<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/78784495-f8fb-4005-a9cb-77598c0d88c8" />

We are still working on this.

Learn **compact, input-dependent reasoning direction vectors** that steer a frozen base LM toward the correct reasoning mode for a given question. The direction is encoded from the question's hidden states, injected as a soft prefix token, and shaped by three auxiliary losses (contrastive InfoNCE, prototype classification, norm regularization) during fine-tuning.

## Technological Setup

**Language:** Python 3 (no version constraint enforced in code).

**Hardware assumptions:**
- CUDA GPU preferred; CPU fallback via `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Mixed-precision training via `torch.amp.autocast` / `GradScaler` (only enabled when CUDA is available)
- Gradient accumulation (default 4 steps) for larger effective batch sizes

**Third-party dependencies (all actually imported across the 10 files):**

| Package | Usage |
|---|---|
| `torch` | Core tensor ops, neural network modules (`nn`), FFT (`nn.functional`), data loading (`utils.data`), AMP (`torch.amp`) |
| `transformers` | `AutoModelForCausalLM`, `AutoTokenizer`, `get_cosine_schedule_with_warmup` |
| `datasets` | `load_dataset` for loading training data (`bench-labs/*`) and evaluation data (`ai2_arc`) |
| `numpy` | Array ops in evaluation metrics and similarity analysis |
| `scikit-learn` (optional) | `KNeighborsClassifier`, `cross_val_score`, `TSNE` (gracefully skipped on `ImportError`) |
| `matplotlib` (optional) | `pyplot` for t-SNE visualizations (gracefully skipped on `ImportError`) |
| `tqdm` | Progress bar in ARC-Easy manual evaluation |
| `lm_eval` (optional) | `lm_eval.simple_evaluate` + `HFLM` wrapper for lm-eval harness evaluation |
| stdlib | `os`, `json`, `copy`, `time`, `argparse`, `functools`, `collections`, `typing`, `types`, `random` |

**Pretrained model checkpoints used:**
- Base LM: `"gpt2"` (OpenAI GPT-2, 124M params, 768 hidden dim, 12 layers, vocab size 50257)
- Training datasets (Hugging Face): `"bench-labs/bench-effortless-6-2026"`, `"bench-labs/bench-easy-6-2026"`, `"bench-labs/bench-mid-6-2026"`
- Evaluation dataset (Hugging Face): `"ai2_arc"` / `"allenai/ai2_arc"` (ARC-Easy, test split)

## Repository Structure

```
code_files/
  ablations.py                 — Runner for ablation experiments (loss ablations, dim scaling, LR sweeps)
  baselines.py                 — Three comparison baselines (category token, standard finetune, learned prefix)
  config.py                    — LRDConfig dataclass: all model, training, loss, and data hyperparameters
  data.py                      — UnifiedDataset + collate_fn: loads 3 bench-labs datasets, normalizes to unified schema
  diagnose.py                  — Diagnostic script to debug ARC-Easy evaluation failures (checkpoint, device, scoring)
  evaluate.py                  — ARC-Easy evaluation (manual + lm-eval wrapper) + direction quality analysis + t-SNE viz
  model.py                     — Core modules: DirectionEncoder, DirectionProjector, LatentDirectionModel, loss functions
  patch_compute_choice_score.py — Standalone monkey-patch for _compute_choice_score on old checkpoints
  run.py                       — Main entry point: CLI dispatcher for train/baselines/evaluate/analyze/ablation/all modes
  train.py                     — Training loop: model setup, optimizer (differential LR), checkpointing, validation
```

## Pipeline Overview

```
run.py (entry point)
 ├── mode="train"     → train.py::train()          → model.py, data.py, config.py
 ├── mode="baselines" → baselines.train_baseline()  → baselines.py, data.py, config.py
 ├── mode="evaluate"  → evaluate.evaluate_arc_easy() + train.load_checkpoint()
 ├── mode="analyze"   → evaluate.analyze_directions() + train.load_checkpoint()
 ├── mode="ablation"  → ablations.run_ablation()    → train.py, config.py
 └── mode="all"       → train → analyze → evaluate  (full pipeline)
```

Cross-file call graph:
- `run.py` imports from `train.py`, `evaluate.py`, `ablations.py`, `baselines.py`
- `train.py` imports from `model.py`, `data.py`, `config.py`
- `evaluate.py` imports from `train.py` (load_checkpoint), `data.py`, `config.py`
- `ablations.py` imports from `train.py`, `config.py`
- `baselines.py` imports from `data.py`, `config.py`
- `diagnose.py` imports from `train.py` (load_checkpoint), `data.py`, `config.py`
- `patch_compute_choice_score.py` imports from `model.py` (apply_patch)

## How to Run

All modes are dispatched through `run.py`:

```bash
# Train the LRD model (default: checkpoints/lrd/)
python run.py --mode train

# Train with custom output directory and config overrides
python run.py --mode train --output_dir checkpoints/my_run --config_override '{"num_epochs": 10, "lr_lm": 5e-5}'

# Train all baselines
python run.py --mode baselines

# Evaluate on ARC-Easy from a checkpoint
python run.py --mode evaluate --checkpoint checkpoints/lrd/checkpoint_best

# Analyze learned directions (quality metrics + optional t-SNE visualization)
python run.py --mode analyze --checkpoint checkpoints/lrd/checkpoint_best --visualize

# Run a single ablation
python run.py --mode ablation --ablation no_contrastive

# Run all ablations
python run.py --mode ablation --ablation all

# Full pipeline: train → analyze → evaluate
python run.py --mode all
```

Individual files can also be run directly:
```bash
python code_files/train.py                    # Train with default config
python code_files/ablations.py --all          # All ablations
python code_files/baselines.py --baseline all # All baselines
python code_files/evaluate.py --checkpoint ... --eval_arc --analyze_directions --visualize
python code_files/diagnose.py --checkpoint checkpoints/lrd/checkpoint_best
python code_files/patch_compute_choice_score.py  # Applies monkey-patch
```

**Key hyperparameters** (see `config.py` `LRDConfig`):
- `model_name`: Base LM (default `"gpt2"`)
- `direction_dim`: Bottleneck dimension for reasoning direction (default 64)
- `alpha_contrastive`, `beta_prototype`, `gamma_norm`: Loss weights
- `lr_lm`: Learning rate for base LM (default 2e-5)
- `lr_direction`: Learning rate for direction modules (default 1e-3)
- `fp16`: Mixed precision (default True, only activates on CUDA)

## Output Structure

```
checkpoints/
  lrd/
    checkpoint_best/    — Best model (base LM HF weights + direction_modules.pt + optimizer.pt)
    checkpoint_final/   — Final model after all epochs
    metrics_log.json    — Per-epoch train/val metrics
  ablations/            — Per-ablation checkpoints
  baselines/            — Per-baseline checkpoints
results/
  arc_easy_results.json — ARC-Easy evaluation accuracy
  ablation_summary.json — Ablation experiment statuses
  baseline_status.json  — Baseline training statuses
analysis/
  direction_quality.json — Direction quality metrics (within/across similarity, k-NN accuracy, collapse)
  directions_tsne.png    — t-SNE visualization (if --visualize)
```

## Environment Variables

None. All paths are relative to the working directory. No external services (wandb, etc.) are configured.

## Data

The three `bench-labs/*` datasets are loaded from Hugging Face. They have differing schemas:
- `effortless`: `{question, answer, category}` (no fine subcategory)
- `easy`: `{question, answer, category}` (with fine subcategory)
- `mid`: `{input, target_scores, category}` (multiple-choice with score dict)

All are normalized to a unified `{question, answer, coarse_category, fine_category}` schema by `data.py:UnifiedDataset`.
