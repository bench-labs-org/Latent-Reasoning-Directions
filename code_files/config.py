"""
Latent Reasoning Direction (LRD) - Configuration

Defines LRDConfig dataclass with all hyperparameters organized by group:
  - Model: model_name (gpt2), hidden_dim, num_layers, vocab_size
  - Direction module: direction_dim, encoder_hidden_dim, num_prefix_tokens
  - Categories: coarse_categories (6), fine_to_coarse mapping (~17 fine → 6 coarse)
  - Training: lr_lm, lr_direction, batch_size, grad_accum_steps, num_epochs,
              warmup_steps, weight_decay, max_grad_norm, max_seq_len, fp16
  - Loss weights: alpha_contrastive, beta_prototype, gamma_norm
  - Temperatures: contrastive_temp, prototype_temp
  - Datasets: dataset_names (3 bench-labs HF datasets)
  - Evaluation: eval_task, eval_num_fewshot
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class LRDConfig:
    # Model
    model_name: str = "gpt2"
    hidden_dim: int = 768
    num_layers: int = 12
    vocab_size: int = 50257

    # Direction module
    direction_dim: int = 64
    encoder_hidden_dim: int = 256
    num_prefix_tokens: int = 1

    # Categories
    coarse_categories: List[str] = field(default_factory=lambda: [
        "Math", "Logic", "Language", "Commonsense", "Knowledge", "PatternRecognition"
    ])

    fine_to_coarse: dict = field(default_factory=lambda: {
        # Math
        "Math": "Math",
        "Math-arithmetic": "Math",
        "Math-reasoning": "Math",
        "Math-pattern": "Math",
        # Logic
        "Logic": "Logic",
        "Logic-consistency": "Logic",
        "Logic-deduction": "Logic",
        "Logic-pattern": "Logic",
        # Language
        "Language": "Language",
        "Language-structure": "Language",
        "Language-transformation": "Language",
        "Language-comprehension": "Language",
        # Commonsense
        "Commonsense": "Commonsense",
        "Commonsense-simulation": "Commonsense",
        "Commonsense-causality": "Commonsense",
        "Commonsense-reasoning": "Commonsense",
        # Knowledge
        "Knowledge": "Knowledge",
        "Knowledge-basic": "Knowledge",
        "Knowledge-definitions": "Knowledge",
        # Pattern Recognition
        "PatternRecognition": "PatternRecognition",
        "Pattern-matching": "PatternRecognition",
        "Pattern-recognition": "PatternRecognition",
        "Pattern-generation": "PatternRecognition",
    })

    # Training
    lr_lm: float = 2e-5
    lr_direction: float = 1e-3
    batch_size: int = 8
    grad_accum_steps: int = 4
    num_epochs: int = 20
    warmup_steps: int = 50
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_len: int = 256
    fp16: bool = True

    # Loss weights
    alpha_contrastive: float = 0.3
    beta_prototype: float = 0.1
    gamma_norm: float = 0.01

    # Temperatures
    contrastive_temp: float = 0.1
    prototype_temp: float = 0.1

    # Datasets
    dataset_names: List[str] = field(default_factory=lambda: [
        "bench-labs/bench-effortless-6-2026",
        "bench-labs/bench-easy-6-2026",
        "bench-labs/bench-mid-6-2026",
    ])

    # Evaluation
    eval_task: str = "arc_easy"
    eval_num_fewshot: int = 0

    @property
    def num_coarse(self) -> int:
        return len(self.coarse_categories)
