"""
Latent Reasoning Direction (LRD) - Main Entry Point

Usage:
    python run.py --mode train                    # Train LRD model
    python run.py --mode train --ablation no_contrastive  # Run ablation
    python run.py --mode baselines                # Train all baselines
    python run.py --mode evaluate --checkpoint checkpoints/lrd/checkpoint_best
    python run.py --mode analyze --checkpoint checkpoints/lrd/checkpoint_best
    python run.py --mode all                      # Full pipeline
"""
import os
import sys
import json
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="Latent Reasoning Direction (LRD)")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "baselines", "evaluate", "analyze", "ablation", "all"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lrd/checkpoint_best")
    parser.add_argument("--output_dir", type=str, default="checkpoints/lrd")
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--config_override", type=str, default=None, 
                        help="JSON string of config overrides")
    args = parser.parse_args()
    
    from config import LRDConfig
    
    # Build config with overrides
    config = LRDConfig()
    if args.config_override:
        overrides = json.loads(args.config_override)
        for key, value in overrides.items():
            setattr(config, key, value)
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("analysis", exist_ok=True)
    
    if args.mode == "train":
        from train import train
        model, tokenizer = train(config, output_dir=args.output_dir)
    
    elif args.mode == "baselines":
        from baselines import train_baseline, CategoryTokenBaseline, StandardFinetuneBaseline, LearnedPrefixBaseline
        
        baselines = {
            "category_token": (CategoryTokenBaseline, "category_token"),
            "standard": (StandardFinetuneBaseline, "standard"),
            "learned_prefix": (LearnedPrefixBaseline, "learned_prefix"),
        }
        
        results = {}
        for cls, name in baselines.items():
            try:
                model, tokenizer = train_baseline(cls, name, config)
                results[name] = "completed"
            except Exception as e:
                results[name] = f"failed: {e}"
        
        with open("results/baseline_status.json", "w") as f:
            json.dump(results, f, indent=2)
    
    elif args.mode == "evaluate":
        from train import load_checkpoint
        from evaluate import evaluate_arc_easy
        
        model, tokenizer, device = load_checkpoint(args.checkpoint)
        results = evaluate_arc_easy(model, tokenizer, device, model.config, args.num_fewshot)
        
        with open("results/arc_easy_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to results/arc_easy_results.json")
    
    elif args.mode == "analyze":
        from train import load_checkpoint
        from evaluate import analyze_directions, visualize_directions
        
        model, tokenizer, device = load_checkpoint(args.checkpoint)
        results, directions, coarse_ids, fine_ids = analyze_directions(model, tokenizer, device, model.config)
        
        with open("analysis/direction_quality.json", "w") as f:
            json.dump({k: float(v) if isinstance(v, (bool,)) else v for k, v in results.items()}, f, indent=2)
        
        if args.visualize:
            visualize_directions(directions, coarse_ids, fine_ids)
    
    elif args.mode == "ablation":
        from ablations import run_ablation, run_all_ablations
        
        ablation_configs = {
            "no_contrastive": {"alpha_contrastive": 0.0},
            "no_prototype": {"beta_prototype": 0.0},
            "no_norm_reg": {"gamma_norm": 0.0},
            "lm_only": {"alpha_contrastive": 0.0, "beta_prototype": 0.0, "gamma_norm": 0.0},
            "dim_8": {"direction_dim": 8},
            "dim_32": {"direction_dim": 32},
            "dim_128": {"direction_dim": 128},
        }
        
        if args.ablation and args.ablation in ablation_configs:
            run_ablation(args.ablation, ablation_configs[args.ablation])
        elif args.ablation == "all":
            run_all_ablations()
        else:
            print(f"Available ablations: {list(ablation_configs.keys())}")
    
    elif args.mode == "all":
        print("=" * 60)
        print("FULL PIPELINE: Train → Analyze → Evaluate")
        print("=" * 60)
        
        # 1. Train
        from train import train, load_checkpoint
        model, tokenizer = train(config, output_dir=args.output_dir)
        
        # 2. Analyze directions
        from evaluate import analyze_directions
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results, directions, coarse_ids, fine_ids = analyze_directions(model, tokenizer, device, config)
        
        with open("analysis/direction_quality.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        if args.visualize:
            from evaluate import visualize_directions
            visualize_directions(directions, coarse_ids, fine_ids)
        
        # 3. Evaluate on ARC-Easy
        from evaluate import evaluate_arc_easy
        eval_results = evaluate_arc_easy(model, tokenizer, device, config, args.num_fewshot)
        
        with open("results/arc_easy_results.json", "w") as f:
            json.dump(eval_results, f, indent=2, default=str)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"  Checkpoints: {args.output_dir}")
        print(f"  Analysis: analysis/direction_quality.json")
        print(f"  Results: results/arc_easy_results.json")


if __name__ == "__main__":
    main()
