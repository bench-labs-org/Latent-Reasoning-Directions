"""
Latent Reasoning Direction (LRD) - Training Script

Handles the full training loop with:
- Mixed precision training (fp16)
- Gradient accumulation
- Differential learning rates (LM vs direction modules)
- Logging and checkpointing
"""
import os
import json
import time
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from config import LRDConfig
from data import create_dataloaders
from model import LatentDirectionModel


def setup_model_and_tokenizer(config: LRDConfig):
    """Initialize model, tokenizer, and direction modules."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    
    model = LatentDirectionModel(base_model, config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, tokenizer, device


def setup_optimizer(model: LatentDirectionModel, config: LRDConfig):
    """Create optimizer with differential learning rates."""
    lm_params = list(model.base_model.parameters())
    direction_params = (
        list(model.direction_encoder.parameters()) +
        list(model.direction_projector.parameters()) +
        [model.prototypes]
    )
    
    param_groups = [
        {"params": lm_params, "lr": config.lr_lm, "name": "lm"},
        {"params": direction_params, "lr": config.lr_direction, "name": "direction"},
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    
    return optimizer


def train(config: LRDConfig = None, output_dir: str = "checkpoints/lrd"):
    """Main training function."""
    if config is None:
        config = LRDConfig()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Latent Reasoning Direction (LRD) Training")
    print("=" * 60)
    
    model, tokenizer, device = setup_model_and_tokenizer(config)
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Direction module parameters: {sum(p.numel() for p in list(model.direction_encoder.parameters()) + list(model.direction_projector.parameters())) + model.prototypes.numel():,}")
    
    optimizer = setup_optimizer(model, config)
    
    train_loader, val_loader, train_dataset = create_dataloaders(config, tokenizer)
    print(f"Train examples: {len(train_loader.dataset)}")
    print(f"Val examples: {len(val_loader.dataset)}")
    print(f"Fine categories: {len(train_dataset.fine_categories)}")
    print(f"Coarse categories: {config.num_coarse}")
    
    # Scheduler
    total_steps = len(train_loader) * config.num_epochs // config.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Mixed precision (use torch.amp API)
    use_amp = config.fp16 and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    
    # Training loop
    global_step = 0
    best_val_loss = float("inf")
    metrics_log = []
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_metrics = {
            "lm_loss": 0, "contrastive_loss": 0, "prototype_loss": 0, 
            "norm_loss": 0, "direction_norm": 0, "count": 0,
        }
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with autocast("cuda", enabled=use_amp):
                loss, metrics = model(batch)
                loss = loss / config.grad_accum_steps
            
            scaler.scale(loss).backward()
            
            # Accumulate metrics
            for key in ["lm_loss", "contrastive_loss", "prototype_loss", "norm_loss", "direction_norm"]:
                epoch_metrics[key] += metrics.get(key, 0.0)
            epoch_metrics["count"] += 1
            
            # Optimizer step
            if (step + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # scheduler.step() AFTER optimizer.step()
                scheduler.step()
                global_step += 1
        
        # Epoch metrics
        n = max(1, epoch_metrics["count"])
        avg_metrics = {k: v / n for k, v in epoch_metrics.items() if k != "count"}
        
        # Validation
        val_metrics = validate(model, val_loader, device, config)
        
        # Logging
        log_entry = {
            "epoch": epoch,
            "global_step": global_step,
            "train": avg_metrics,
            "val": val_metrics,
            "lr_lm": optimizer.param_groups[0]["lr"],
            "lr_direction": optimizer.param_groups[1]["lr"],
        }
        metrics_log.append(log_entry)
        
        print(f"Epoch {epoch+1}/{config.num_epochs} | "
              f"Train LM: {avg_metrics['lm_loss']:.4f} | "
              f"Train Con: {avg_metrics['contrastive_loss']:.4f} | "
              f"Val LM: {val_metrics['lm_loss']:.4f} | "
              f"Val Total: {val_metrics['total_loss']:.4f} | "
              f"Dir Norm: {avg_metrics['direction_norm']:.3f}")
        
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            save_checkpoint(model, optimizer, config, output_dir, epoch, "best")
            print(f"  → New best model saved (val_loss={best_val_loss:.4f})")
        
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, config, output_dir, epoch, f"epoch_{epoch+1}")
    
    save_checkpoint(model, optimizer, config, output_dir, config.num_epochs - 1, "final")
    
    with open(os.path.join(output_dir, "metrics_log.json"), "w") as f:
        json.dump(metrics_log, f, indent=2)
    
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    
    return model, tokenizer


@torch.no_grad()
def validate(model: LatentDirectionModel, val_loader, device, config: LRDConfig) -> dict:
    """Run validation and return metrics."""
    model.eval()
    val_metrics = {
        "total_loss": 0, "lm_loss": 0, "contrastive_loss": 0, 
        "prototype_loss": 0, "norm_loss": 0, "direction_norm": 0, "count": 0,
    }
    
    use_amp = config.fp16 and device.type == "cuda"
    
    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with autocast("cuda", enabled=use_amp):
            loss, metrics = model(batch)
        
        val_metrics["total_loss"] += metrics["loss"]
        for key in ["lm_loss", "contrastive_loss", "prototype_loss", "norm_loss", "direction_norm"]:
            val_metrics[key] += metrics.get(key, 0.0)
        val_metrics["count"] += 1
    
    n = max(1, val_metrics["count"])
    avg = {k: v / n for k, v in val_metrics.items() if k != "count"}
    model.train()
    return avg


def save_checkpoint(model, optimizer, config, output_dir, epoch, tag):
    """Save model checkpoint."""
    path = os.path.join(output_dir, f"checkpoint_{tag}")
    os.makedirs(path, exist_ok=True)
    
    # Save base model
    model.base_model.save_pretrained(path)
    
    # Save tokenizer (critical!)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(path)
    
    # Save direction modules
    torch.save({
        "direction_encoder": model.direction_encoder.state_dict(),
        "direction_projector": model.direction_projector.state_dict(),
        "prototypes": model.prototypes.data,
        "config": config,
        "epoch": epoch,
    }, os.path.join(path, "direction_modules.pt"))
    
    # Save optimizer
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))


def load_checkpoint(path: str, config: LRDConfig = None):
    """Load a checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_model = AutoModelForCausalLM.from_pretrained(path)
    
    # Load tokenizer - try checkpoint first, fall back to original model
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.vocab_size == 0:
        # Tokenizer wasn't saved properly, load from original model
        modules = torch.load(os.path.join(path, "direction_modules.pt"), 
                            map_location=device, weights_only=False)
        cfg = modules["config"]
        model_name = getattr(cfg, 'model_name', 'gpt2')
        print(f"   ⚠️  Tokenizer missing from checkpoint, loading from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if config is None:
        modules = torch.load(os.path.join(path, "direction_modules.pt"), 
                            map_location=device, weights_only=False)
        config = modules["config"]
    
    model = LatentDirectionModel(base_model, config)
    
    modules = torch.load(os.path.join(path, "direction_modules.pt"), 
                        map_location=device, weights_only=False)
    model.direction_encoder.load_state_dict(modules["direction_encoder"])
    model.direction_projector.load_state_dict(modules["direction_projector"])
    model.prototypes.data = modules["prototypes"]
    
    model = model.to(device)
    return model, tokenizer, device


if __name__ == "__main__":
    config = LRDConfig()
    model, tokenizer = train(config)
