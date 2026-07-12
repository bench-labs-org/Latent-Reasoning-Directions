"""
Latent Reasoning Direction (LRD) - Baseline Models

Implements alternative approaches for comparison:
1. CategoryTokenBaseline: Special category tokens prepended to input
2. StandardFinetuneBaseline: Standard LM fine-tuning with loss masking only
3. LearnedPrefixBaseline: Random-init learned prefix without direction supervision
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from config import LRDConfig
from data import create_dataloaders


class CategoryTokenBaseline(nn.Module):
    """
    Baseline: Add 6 special tokens (<|Math|>, <|Logic|>, etc.) to vocabulary.
    Prepend the appropriate token to each example.
    
    The token embedding IS the direction - learned end-to-end.
    """
    
    def __init__(self, base_model, config: LRDConfig, tokenizer):
        super().__init__()
        self.config = config
        self.base_model = base_model
        self.tokenizer = tokenizer
        
        # Add special tokens
        self.category_tokens = [f"<|{cat}|>" for cat in config.coarse_categories]
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": self.category_tokens})
        base_model.resize_token_embeddings(len(tokenizer))
        
        self.category_token_ids = [
            tokenizer.convert_tokens_to_ids(t) for t in self.category_tokens
        ]
    
    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss_mask = batch["loss_mask"]
        coarse_ids = batch["coarse_category_ids"]
        num_prefix = batch["num_prefix"]
        
        B = input_ids.shape[0]
        device = input_ids.device
        
        # Replace prefix position with category token ID
        for i in range(B):
            cat_token_id = self.category_token_ids[coarse_ids[i].item()]
            input_ids[i, 0] = cat_token_id
        
        # Forward
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        
        # Masked LM loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = loss_mask[:, 1:].contiguous()
        
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none',
        ).view(B, -1)
        
        lm_loss = (per_token_loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)
        
        return lm_loss, {"loss": lm_loss.item(), "lm_loss": lm_loss.item()}


class StandardFinetuneBaseline(nn.Module):
    """
    Baseline: Standard LM fine-tuning with loss masking only.
    No direction prefix, no auxiliary losses.
    Tests whether the model implicitly learns reasoning modes.
    """
    
    def __init__(self, base_model, config: LRDConfig):
        super().__init__()
        self.config = config
        self.base_model = base_model
    
    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss_mask = batch["loss_mask"]
        num_prefix = batch["num_prefix"]
        
        B = input_ids.shape[0]
        
        # Remove prefix position (not used in this baseline)
        input_ids = input_ids[:, num_prefix:]
        attention_mask = attention_mask[:, num_prefix:]
        loss_mask = loss_mask[:, num_prefix:]
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = loss_mask[:, 1:].contiguous()
        
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none',
        ).view(B, -1)
        
        lm_loss = (per_token_loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)
        
        return lm_loss, {"loss": lm_loss.item(), "lm_loss": lm_loss.item()}


class LearnedPrefixBaseline(nn.Module):
    """
    Baseline: A learned prefix vector (random init) without direction supervision.
    The prefix is a fixed learnable parameter, not conditioned on the input.
    Tests whether any prefix helps, regardless of direction.
    """
    
    def __init__(self, base_model, config: LRDConfig):
        super().__init__()
        self.config = config
        self.base_model = base_model
        
        # Learnable prefix (not conditioned on anything)
        self.prefix = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
    
    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss_mask = batch["loss_mask"]
        num_prefix = batch["num_prefix"]
        
        B = input_ids.shape[0]
        
        # Get embeddings
        embeds = self.base_model.get_input_embeddings()(input_ids)
        
        # Replace prefix with learned prefix
        prefix_expanded = self.prefix.expand(B, -1, -1)
        embeds[:, :num_prefix, :] = prefix_expanded
        
        outputs = self.base_model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = loss_mask[:, 1:].contiguous()
        
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none',
        ).view(B, -1)
        
        lm_loss = (per_token_loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)
        
        return lm_loss, {"loss": lm_loss.item(), "lm_loss": lm_loss.item()}


def train_baseline(model_class, name, config=None, output_dir=None):
    """Train a baseline model."""
    if config is None:
        config = LRDConfig()
    if output_dir is None:
        output_dir = f"checkpoints/baselines/{name}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training Baseline: {name}")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    
    # Instantiate baseline
    if name == "category_token":
        model = model_class(base_model, config, tokenizer)
    else:
        model = model_class(base_model, config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Dataloader
    train_loader, val_loader, _ = create_dataloaders(config, tokenizer)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_lm, weight_decay=config.weight_decay)
    
    total_steps = len(train_loader) * config.num_epochs // config.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)
    
    scaler = GradScaler(enabled=config.fp16)
    
    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        count = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with autocast(enabled=config.fp16):
                loss, metrics = model(batch)
                loss = loss / config.grad_accum_steps
            
            scaler.scale(loss).backward()
            
            epoch_loss += metrics["lm_loss"]
            count += 1
            
            if (step + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        
        avg_loss = epoch_loss / max(1, count)
        
        # Validation
        model.eval()
        val_loss = 0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                _, metrics = model(batch)
                val_loss += metrics["lm_loss"]
                val_count += 1
        
        avg_val_loss = val_loss / max(1, val_count)
        
        print(f"  Epoch {epoch+1}/{config.num_epochs} | Train: {avg_loss:.4f} | Val: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.base_model.save_pretrained(os.path.join(output_dir, "best"))
            tokenizer.save_pretrained(os.path.join(output_dir, "best"))
    
    # Save final
    model.base_model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    
    print(f"  Best val loss: {best_val_loss:.4f}")
    return model, tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="all", 
                        choices=["all", "category_token", "standard", "learned_prefix"])
    args = parser.parse_args()
    
    baselines = {
        "category_token": (CategoryTokenBaseline, "category_token"),
        "standard": (StandardFinetuneBaseline, "standard"),
        "learned_prefix": (LearnedPrefixBaseline, "learned_prefix"),
    }
    
    if args.baseline == "all":
        for cls, name in baselines.values():
            train_baseline(cls, name)
    else:
        cls, name = baselines[args.baseline]
        train_baseline(cls, name)
