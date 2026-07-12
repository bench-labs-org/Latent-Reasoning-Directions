"""
Latent Reasoning Direction (LRD) - Model Modules

Contains:
- DirectionEncoder: Maps pooled hidden states to compact reasoning direction
- DirectionProjector: Projects direction back to LM hidden dimension
- LatentDirectionModel: Full model wrapping base LM + direction modules
- Loss functions: InfoNCE, Prototype, Norm regularization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from config import LRDConfig


class DirectionEncoder(nn.Module):
    """
    Maps pooled hidden states from the base LM to a compact reasoning direction.
    
    Architecture: Linear(H→256) → GELU → Linear(256→D) → Tanh
    
    The bottleneck (D << H) prevents the direction from encoding answer details.
    The Tanh activation bounds the output for stability.
    """
    
    def __init__(self, hidden_dim: int = 768, direction_dim: int = 64, 
                 encoder_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, encoder_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden, direction_dim),
            nn.Tanh(),
        )
    
    def forward(self, h_pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_pooled: [B, H] mean-pooled hidden states from question
        
        Returns:
            direction: [B, D] bounded direction vector
        """
        return self.net(h_pooled)


class DirectionProjector(nn.Module):
    """
    Projects the direction vector back to LM hidden dimension for prefix injection.
    
    This creates the soft prefix token that will be prepended to the input sequence.
    """
    
    def __init__(self, direction_dim: int = 64, hidden_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(direction_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            d: [B, D] normalized direction vector
        
        Returns:
            prefix: [B, H] prefix hidden state
        """
        return self.proj(d)


class LatentDirectionModel(nn.Module):
    """
    Full LRD model: Base LM + Direction Encoder + Direction Projector + Prototypes.
    
    Forward pass:
    1. Encode question through base LM → hidden states
    2. Pool hidden states → Direction Encoder → compact direction d ∈ R^D
    3. Project d → prefix hidden state p ∈ R^H
    4. Concatenate [p, question_embeds, answer_embeds]
    5. Forward through LM with inputs_embeds
    6. Compute masked LM loss + auxiliary losses
    """
    
    def __init__(self, base_model, config: LRDConfig):
        super().__init__()
        self.config = config
        self.base_model = base_model
        
        # Direction modules
        self.direction_encoder = DirectionEncoder(
            hidden_dim=config.hidden_dim,
            direction_dim=config.direction_dim,
            encoder_hidden=config.encoder_hidden_dim,
        )
        self.direction_projector = DirectionProjector(
            direction_dim=config.direction_dim,
            hidden_dim=config.hidden_dim,
        )
        
        # Learnable prototypes for coarse categories
        self.prototypes = nn.Parameter(
            torch.randn(config.num_coarse, config.direction_dim) * 0.1
        )
    
    def encode_direction(self, question_ids: torch.Tensor, 
                         question_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the reasoning direction from question tokens.
        
        Args:
            question_ids: [B, L_q] question token IDs
            question_mask: [B, L_q] attention mask
        
        Returns:
            d_normalized: [B, D] normalized direction vector
        """
        # Get question hidden states
        with torch.no_grad():
            q_outputs = self.base_model(
                input_ids=question_ids,
                attention_mask=question_mask,
                output_hidden_states=True,
            )
            # CausalLM models return hidden_states as a tuple of all layers
            q_hidden = q_outputs.hidden_states[-1]  # [B, L_q, H]
        
        # Mean pool over non-padding positions
        mask_expanded = question_mask.unsqueeze(-1).float()  # [B, L_q, 1]
        h_pooled = (q_hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        
        # Encode direction
        d = self.direction_encoder(h_pooled)  # [B, D]
        d_normalized = F.normalize(d, dim=-1)
        
        return d_normalized
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Full forward pass with all losses.
        
        Args:
            batch: Dict with keys from collate_fn
        
        Returns:
            total_loss: scalar
            metrics: dict with component losses and directions
        """
        input_ids = batch["input_ids"]           # [B, L_total]
        attention_mask = batch["attention_mask"]  # [B, L_total]
        loss_mask = batch["loss_mask"]           # [B, L_total]
        coarse_ids = batch["coarse_category_ids"]  # [B]
        fine_ids = batch["fine_category_ids"]     # [B]
        q_lens = batch["question_lens"]           # [B]
        num_prefix = batch["num_prefix"]          # scalar
        
        B = input_ids.shape[0]
        device = input_ids.device
        
        # Extract question portion for direction encoding
        # Question starts after the prefix position
        q_start = num_prefix
        q_end = num_prefix + q_lens.max().item()
        q_ids = input_ids[:, q_start:q_end]
        q_mask = attention_mask[:, q_start:q_end]
        
        # Compute direction (with gradient flow through encoder)
        # Note: we use no_grad for the base LM forward in encode_direction,
        # but the direction_encoder IS trained
        with torch.no_grad():
            q_outputs = self.base_model(input_ids=q_ids, attention_mask=q_mask, output_hidden_states=True)
            q_hidden = q_outputs.hidden_states[-1]
        
        mask_expanded = q_mask.unsqueeze(-1).float()
        h_pooled = (q_hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        
        # Direction encoding (WITH gradients)
        d = self.direction_encoder(h_pooled)  # [B, D]
        d_normalized = F.normalize(d, dim=-1)
        
        # Project to prefix hidden state
        prefix_hidden = self.direction_projector(d_normalized)  # [B, H]
        
        # Get embeddings for all tokens
        embed_layer = self.base_model.get_input_embeddings()
        all_embeds = embed_layer(input_ids)  # [B, L_total, H]
        
        # Replace prefix position(s) with projected direction
        all_embeds[:, :num_prefix, :] = prefix_hidden.unsqueeze(1)
        
        # Forward through LM with modified embeddings
        outputs = self.base_model(
            inputs_embeds=all_embeds,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # [B, L_total, V]
        
        # ===== LM Loss (masked) =====
        # Shift for autoregressive prediction
        shift_logits = logits[:, :-1, :].contiguous()    # [B, L-1, V]
        shift_labels = input_ids[:, 1:].contiguous()      # [B, L-1]
        shift_mask = loss_mask[:, 1:].contiguous()        # [B, L-1]
        
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none',
        ).view(B, -1)
        
        # Apply mask: only answer tokens contribute
        masked_loss = per_token_loss * shift_mask
        lm_loss = masked_loss.sum() / shift_mask.sum().clamp(min=1)
        
        # ===== Contrastive Loss (InfoNCE) =====
        contrastive_loss = self._infonce_loss(d_normalized, fine_ids)
        
        # ===== Prototype Loss =====
        proto_loss = self._prototype_loss(d_normalized, coarse_ids)
        
        # ===== Norm Regularization =====
        # Encourage the raw direction to have a stable, non-trivial norm
        # (prevents collapse to zero before normalization).
        # Target norm = sqrt(D) * 0.5 (roughly the expected norm of tanh outputs)
        target_norm = (self.config.direction_dim ** 0.5) * 0.5
        norm_loss = ((d.norm(dim=-1) - target_norm) ** 2).mean()
        
        # ===== Combined Loss =====
        total_loss = (
            lm_loss 
            + self.config.alpha_contrastive * contrastive_loss
            + self.config.beta_prototype * proto_loss
            + self.config.gamma_norm * norm_loss
        )
        
        metrics = {
            "loss": total_loss.item(),
            "lm_loss": lm_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "prototype_loss": proto_loss.item(),
            "norm_loss": norm_loss.item(),
            "directions": d_normalized.detach(),
            "direction_norm": d_normalized.norm(dim=-1).mean().item(),
        }
        
        return total_loss, metrics
    
    def _infonce_loss(self, directions: torch.Tensor, labels: torch.Tensor, 
                      temperature: Optional[float] = None) -> torch.Tensor:
        """
        InfoNCE contrastive loss.
        
        Pulls together directions from the same fine category,
        pushes apart directions from different categories.
        """
        if temperature is None:
            temperature = self.config.contrastive_temp
        
        B = directions.shape[0]
        
        if B < 2:
            return torch.tensor(0.0, device=directions.device)
        
        # Cosine similarity matrix (directions are already normalized)
        sim = directions @ directions.T / temperature  # [B, B]
        
        # Positive mask: same fine category, excluding self
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)
        
        # If no positive pairs in batch, return 0
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=directions.device)
        
        # Mask self-similarity
        sim = sim - torch.eye(B, device=directions.device) * 1e9
        
        # Log-softmax over all non-self examples
        log_probs = F.log_softmax(sim, dim=1)  # [B, B]
        
        # Average log-prob over positive pairs
        pos_count = pos_mask.sum(dim=1).clamp(min=1)
        loss = -(log_probs * pos_mask).sum(dim=1) / pos_count
        
        # Only include examples that have at least one positive pair
        has_positive = pos_mask.sum(dim=1) > 0
        if has_positive.sum() == 0:
            return torch.tensor(0.0, device=directions.device)
        
        return loss[has_positive].mean()
    
    def _prototype_loss(self, directions: torch.Tensor, 
                        coarse_labels: torch.Tensor) -> torch.Tensor:
        """
        Prototype classification loss.
        
        Each direction should be close to its coarse category prototype.
        """
        # Normalize prototypes
        proto_norm = F.normalize(self.prototypes, dim=-1)  # [C, D]
        
        # Cosine similarity
        logits = directions @ proto_norm.T / self.config.prototype_temp  # [B, C]
        
        return F.cross_entropy(logits, coarse_labels)
    
    @torch.no_grad()
    def generate_with_direction(self, question: str, tokenizer, 
                                 max_new_tokens: int = 50, **gen_kwargs) -> str:
        """
        Generate an answer for a question using the computed direction.
        
        Used for evaluation/inference.
        """
        self.eval()
        
        # Tokenize question
        q_tok = tokenizer(question, return_tensors="pt", truncation=True, 
                          max_length=self.config.max_seq_len // 2)
        q_ids = q_tok["input_ids"].to(self.prototypes.device)
        q_mask = q_tok["attention_mask"].to(self.prototypes.device)
        
        # Compute direction
        q_outputs = self.base_model(input_ids=q_ids, attention_mask=q_mask, output_hidden_states=True)
        q_hidden = q_outputs.hidden_states[-1]
        mask_exp = q_mask.unsqueeze(-1).float()
        h_pooled = (q_hidden * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        
        d = self.direction_encoder(h_pooled)
        d_normalized = F.normalize(d, dim=-1)
        
        # Project to prefix
        prefix = self.direction_projector(d_normalized).unsqueeze(1)  # [1, 1, H]
        
        # Get question embeddings
        embed_layer = self.base_model.get_input_embeddings()
        q_embeds = embed_layer(q_ids)  # [1, L_q, H]
        
        # Concatenate
        full_embeds = torch.cat([prefix, q_embeds], dim=1)
        full_mask = torch.ones(1, full_embeds.shape[1], device=full_embeds.device, dtype=torch.long)
        
        # Generate
        output = self.base_model.generate(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **gen_kwargs,
        )
        
        return tokenizer.decode(output[0], skip_special_tokens=True)

    @torch.no_grad()
    def _compute_choice_score(self, question: str, choice: str, 
                               tokenizer, device) -> float:
        """
        Score a (question, choice) pair by computing the average log-likelihood
        of the choice tokens given the question with direction prefix injected.
        
        Used for multiple-choice evaluation (e.g., ARC-Easy).
        
        Returns -inf if tokenization fails.
        """
        self.eval()
        
        max_q_len = getattr(self.config, 'max_seq_len', 256) // 2
        max_c_len = getattr(self.config, 'max_seq_len', 256) // 4
        
        # Tokenize question and choice
        q_tok = tokenizer(question, return_tensors="pt", truncation=True, 
                          max_length=max_q_len)
        c_tok = tokenizer(choice, return_tensors="pt", truncation=True, 
                          max_length=max_c_len)
        
        q_ids = q_tok["input_ids"].to(device)
        q_mask = q_tok["attention_mask"].to(device)
        c_ids = c_tok["input_ids"].to(device)
        
        # Guard against empty tokenization
        if q_ids.shape[1] == 0 or c_ids.shape[1] == 0:
            return float("-inf")
        
        # Compute direction from question
        q_outputs = self.base_model(input_ids=q_ids, attention_mask=q_mask, 
                                     output_hidden_states=True)
        q_hidden = q_outputs.hidden_states[-1]
        mask_exp = q_mask.unsqueeze(-1).float()
        h_pooled = (q_hidden * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        
        d = self.direction_encoder(h_pooled)
        d_normalized = F.normalize(d, dim=-1)
        
        # Project to prefix
        prefix = self.direction_projector(d_normalized).unsqueeze(1)
        
        # Concatenate: [prefix, question, choice]
        embed_layer = self.base_model.get_input_embeddings()
        q_embeds = embed_layer(q_ids)
        c_embeds = embed_layer(c_ids)
        
        full_embeds = torch.cat([prefix, q_embeds, c_embeds], dim=1)
        full_mask = torch.ones(1, full_embeds.shape[1], device=device, dtype=torch.long)
        
        # Forward pass
        outputs = self.base_model(inputs_embeds=full_embeds, attention_mask=full_mask)
        logits = outputs.logits
        
        # Compute log-likelihood of choice tokens
        L_q = q_ids.shape[1]
        L_c = c_ids.shape[1]
        
        choice_logits = logits[0, L_q:L_q + L_c, :]
        choice_targets = c_ids[0, :]
        
        log_probs = F.log_softmax(choice_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, choice_targets.unsqueeze(-1)).squeeze(-1)
        
        # Length-normalized score
        score = token_log_probs.sum().item() / max(1, L_c)
        
        return score
