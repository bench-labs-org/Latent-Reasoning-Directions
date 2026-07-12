"""
Latent Reasoning Direction (LRD) - Ablation Runner

Runs all planned ablation experiments and collects results.
"""
import os
import json
import copy
import torch
from config import LRDConfig
from train import train


def run_ablation(name: str, config_overrides: dict, output_base: str = "checkpoints/ablations"):
    """Run a single ablation experiment."""
    config = LRDConfig()
    for key, value in config_overrides.items():
        setattr(config, key, value)
    
    output_dir = os.path.join(output_base, name)
    print(f"\n{'='*60}")
    print(f"ABLATION: {name}")
    print(f"{'='*60}")
    print(f"Config overrides: {config_overrides}")
    
    model, tokenizer = train(config, output_dir=output_dir)
    
    return model, tokenizer, output_dir


def run_all_ablations():
    """Run all planned ablation experiments."""
    
    ablations = {
        # A1: Direction necessity
        "no_direction": {
            "alpha_contrastive": 0.0,
            "beta_prototype": 0.0,
            "gamma_norm": 0.0,
            # Note: direction prefix is still injected but losses don't shape it
        },
        
        # A2: Loss component ablations
        "no_contrastive": {"alpha_contrastive": 0.0},
        "no_prototype": {"beta_prototype": 0.0},
        "no_norm_reg": {"gamma_norm": 0.0},
        "lm_only": {
            "alpha_contrastive": 0.0,
            "beta_prototype": 0.0,
            "gamma_norm": 0.0,
        },
        
        # A3: Direction dimensionality
        "dim_8": {"direction_dim": 8},
        "dim_32": {"direction_dim": 32},
        "dim_128": {"direction_dim": 128},
        "dim_256": {"direction_dim": 256},
        
        # A4: Training hyperparameters
        "low_lr": {"lr_lm": 1e-5, "lr_direction": 5e-4},
        "high_lr": {"lr_lm": 5e-5, "lr_direction": 2e-3},
        "fewer_epochs": {"num_epochs": 10},
        "more_epochs": {"num_epochs": 30},
    }
    
    results = {}
    
    for name, overrides in ablations.items():
        try:
            model, tokenizer, output_dir = run_ablation(name, overrides)
            results[name] = {"status": "completed", "output_dir": output_dir}
        except Exception as e:
            results[name] = {"status": "failed", "error": str(e)}
            print(f"  FAILED: {e}")
    
    # Save ablation summary
    os.makedirs("results", exist_ok=True)
    with open("results/ablation_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Ablation Summary")
    print(f"{'='*60}")
    for name, result in results.items():
        status = result["status"]
        print(f"  {name}: {status}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, default=None, help="Run specific ablation")
    parser.add_argument("--all", action="store_true", help="Run all ablations")
    args = parser.parse_args()
    
    if args.all:
        run_all_ablations()
    elif args.ablation:
        # Map name to overrides
        ablation_configs = {
            "no_direction": {"alpha_contrastive": 0.0, "beta_prototype": 0.0, "gamma_norm": 0.0},
            "no_contrastive": {"alpha_contrastive": 0.0},
            "no_prototype": {"beta_prototype": 0.0},
            "lm_only": {"alpha_contrastive": 0.0, "beta_prototype": 0.0, "gamma_norm": 0.0},
            "dim_8": {"direction_dim": 8},
            "dim_32": {"direction_dim": 32},
            "dim_128": {"direction_dim": 128},
        }
        if args.ablation in ablation_configs:
            run_ablation(args.ablation, ablation_configs[args.ablation])
        else:
            print(f"Unknown ablation: {args.ablation}")
            print(f"Available: {list(ablation_configs.keys())}")
    else:
        print("Use --all to run all ablations or --ablation <name> for a specific one")
