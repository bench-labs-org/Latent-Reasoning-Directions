"""
Standalone _compute_choice_score function for monkey-patching old checkpoints
that were saved before the method was added to LatentDirectionModel.

Usage:
    from patch_compute_choice_score import apply_patch
    apply_patch()  # Patches LatentDirectionModel._compute_choice_score in-place

Also callable directly via __main__:
    python patch_compute_choice_score.py
"""
import torch
import torch.nn.functional as F

def _compute_choice_score(self, question: str, choice: str, tokenizer, device) -> float:
    """
    Score a (question, choice) pair by computing the average log-likelihood
    of the choice tokens given the question with direction prefix injected.
    
    Returns -inf if tokenization fails.
    """
    self.eval()
    
    max_q_len = getattr(self.config, 'max_seq_len', 256) // 2
    max_c_len = getattr(self.config, 'max_seq_len', 256) // 4
    
    # Tokenize question
    q_tok = tokenizer(question, return_tensors="pt", truncation=True, 
                      max_length=max_q_len)
    # Tokenize choice (add special tokens to avoid empty sequences)
    c_tok = tokenizer(choice, return_tensors="pt", truncation=True, 
                      max_length=max_c_len)
    
    q_ids = q_tok["input_ids"].to(device)
    q_mask = q_tok["attention_mask"].to(device)
    c_ids = c_tok["input_ids"].to(device)
    
    # Guard against empty tokenization
    if q_ids.shape[1] == 0 or c_ids.shape[1] == 0:
        return float("-inf")
    
    with torch.no_grad():
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
        
        # choice_logits: logits at positions [L_q, L_q+L_c) predict tokens at [L_q+1, L_q+L_c+1)
        # But our targets are c_ids at positions [0, L_c)
        # The prefix is at position 0, question at [1, L_q+1), choice at [L_q+1, L_q+L_c+1)
        # So logits at position L_q predict token at position L_q+1 (first choice token)
        choice_logits = logits[0, L_q:L_q + L_c, :]    # [L_c, V]
        choice_targets = c_ids[0, :]                     # [L_c]
        
        log_probs = F.log_softmax(choice_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, choice_targets.unsqueeze(-1)).squeeze(-1)
        
        # Length-normalized score
        score = token_log_probs.sum().item() / max(1, L_c)
    
    return score

def apply_patch():
    """Apply the patch to LatentDirectionModel."""
    from model import LatentDirectionModel
    LatentDirectionModel._compute_choice_score = _compute_choice_score
    print("✅ Patched LatentDirectionModel with _compute_choice_score method")

if __name__ == "__main__":
    apply_patch()
