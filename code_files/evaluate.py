"""
Latent Reasoning Direction (LRD) - Evaluation Script

Evaluates trained LRD model on ARC-Easy via lm-eval.
Also provides direction quality analysis.
"""
import os
import sys
import json
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from config import LRDConfig
from train import load_checkpoint
from data import UnifiedDataset
from transformers import AutoTokenizer


# ============================================================================
# ARC-Easy Evaluation
# ============================================================================

def evaluate_arc_easy(model, tokenizer, device, config, num_fewshot=0):
    """
    Evaluate on ARC-Easy using manual evaluation.
    """
    # Apply monkey patch if method is missing (for old checkpoints)
    if not hasattr(model, '_compute_choice_score'):
        print("Patching model with _compute_choice_score method...")
        import types
        
        def _compute_choice_score(self, question, choice, tokenizer, device):
            """Score a (question, choice) pair with direction injection."""
            import torch
            import torch.nn.functional as F
            self.eval()
            max_q = getattr(self.config, 'max_seq_len', 256) // 2
            max_c = getattr(self.config, 'max_seq_len', 256) // 4
            q_tok = tokenizer(question, return_tensors="pt", truncation=True, max_length=max_q)
            c_tok = tokenizer(choice, return_tensors="pt", truncation=True, max_length=max_c)
            q_ids = q_tok["input_ids"].to(device)
            q_mask = q_tok["attention_mask"].to(device)
            c_ids = c_tok["input_ids"].to(device)
            if q_ids.shape[1] == 0 or c_ids.shape[1] == 0:
                return float("-inf")
            with torch.no_grad():
                q_out = self.base_model(input_ids=q_ids, attention_mask=q_mask, output_hidden_states=True)
                q_h = q_out.hidden_states[-1]
                me = q_mask.unsqueeze(-1).float()
                hp = (q_h * me).sum(1) / me.sum(1).clamp(min=1)
                d = F.normalize(self.direction_encoder(hp), dim=-1)
                pfx = self.direction_projector(d).unsqueeze(1)
                emb = self.base_model.get_input_embeddings()
                full = torch.cat([pfx, emb(q_ids), emb(c_ids)], dim=1)
                fm = torch.ones(1, full.shape[1], device=device, dtype=torch.long)
                out = self.base_model(inputs_embeds=full, attention_mask=fm)
                logits = out.logits
                L_q, L_c = q_ids.shape[1], c_ids.shape[1]
                cl = logits[0, L_q:L_q+L_c, :]
                ct = c_ids[0, :]
                lp = F.log_softmax(cl, dim=-1)
                tlp = lp.gather(-1, ct.unsqueeze(-1)).squeeze(-1)
                return tlp.sum().item() / max(1, L_c)
        
        model._compute_choice_score = types.MethodType(_compute_choice_score, model)
        print("✅ Patch applied successfully")
    
    return manual_arc_eval(model, tokenizer, device, config)


def evaluate_via_lm_eval(model, tokenizer, device, config, num_fewshot=0):
    """Evaluate using lm-eval library."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    
    class LRDWrapper(HFLM):
        """Wrapper that injects direction prefix during lm-eval inference."""
        
        def __init__(self, lrd_model, tokenizer, device, lrd_config):
            self.lrd_model = lrd_model
            self._tokenizer = tokenizer
            self._device = device
            self.lrd_config = lrd_config
            self.lrd_model.eval()
        
        @property
        def tokenizer(self):
            return self._tokenizer
        
        @property
        def device(self):
            return self._device
        
        @property
        def model(self):
            return self.lrd_model.base_model
        
        def loglikelihood(self, requests, disable_tqdm=False):
            results = []
            for context, continuation in [r.args for r in requests]:
                ll, is_greedy = self._compute_loglikelihood(context, continuation)
                results.append((ll, is_greedy))
            return results
        
        def _compute_loglikelihood(self, context, continuation):
            with torch.no_grad():
                full_text = context + continuation
                full_tokens = self._tokenizer(full_text, return_tensors="pt", 
                                              truncation=True, max_length=512)
                ctx_tokens = self._tokenizer(context, return_tensors="pt",
                                             truncation=True, max_length=256)
                
                input_ids = full_tokens["input_ids"].to(self._device)
                ctx_ids = ctx_tokens["input_ids"].to(self._device)
                ctx_mask = ctx_tokens["attention_mask"].to(self._device)
                
                # Compute direction from context
                ctx_outputs = self.lrd_model.base_model(
                    input_ids=ctx_ids, attention_mask=ctx_mask, output_hidden_states=True
                )
                ctx_hidden = ctx_outputs.hidden_states[-1]
                mask_exp = ctx_mask.unsqueeze(-1).float()
                h_pooled = (ctx_hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)
                
                d = self.lrd_model.direction_encoder(h_pooled)
                d_normalized = F.normalize(d, dim=-1)
                prefix = self.lrd_model.direction_projector(d_normalized).unsqueeze(1)
                
                # Get embeddings and inject prefix
                embeds = self.lrd_model.base_model.get_input_embeddings()(input_ids)
                full_embeds = torch.cat([prefix, embeds], dim=1)
                full_mask = torch.ones(1, full_embeds.shape[1], device=self._device, dtype=torch.long)
                
                outputs = self.lrd_model.base_model(
                    inputs_embeds=full_embeds, attention_mask=full_mask
                )
                logits = outputs.logits
                
                # Compute log-likelihood of continuation
                cont_start = ctx_ids.shape[1]
                log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
                target_ids = input_ids[0, 1:]
                target_log_probs = log_probs[0, 1:]
                
                cont_len = input_ids.shape[1] - ctx_ids.shape[1]
                cont_positions = slice(target_log_probs.shape[0] - cont_len, target_log_probs.shape[0])
                
                cont_log_probs = target_log_probs[cont_positions]
                cont_target = target_ids[-cont_len:]
                
                ll = cont_log_probs.gather(-1, cont_target.unsqueeze(-1)).squeeze(-1).sum().item()
                greedy_tokens = cont_log_probs.argmax(dim=-1)
                is_greedy = (greedy_tokens == cont_target).all().item()
                
                return ll, is_greedy
    
    wrapper = LRDWrapper(model, tokenizer, device, config)
    
    results = lm_eval.simple_evaluate(
        model=wrapper,
        tasks=["arc_easy"],
        num_fewshot=num_fewshot,
        batch_size=1,
    )
    
    return results


def manual_arc_eval(model, tokenizer, device, config):
    """
    Manual ARC-Easy evaluation without lm-eval.
    Downloads ARC-Easy and evaluates directly using _compute_choice_score.
    """
    from datasets import load_dataset
    from tqdm import tqdm
    
    try:
        arc_ds = load_dataset("ai2_arc", "ARC-Easy")
    except Exception:
        arc_ds = load_dataset("allenai/ai2_arc", "ARC-Easy")
    
    split = "test"
    correct = 0
    total = 0
    
    model.eval()
    errors = 0
    
    for example in tqdm(arc_ds[split], desc="Evaluating ARC-Easy"):
        question = example["question"]
        choices = example["choices"]
        answer_key = example["answerKey"]
        
        choice_labels = choices["label"]
        choice_texts = choices["text"]
        
        if not question or not choice_texts:
            errors += 1
            continue
        
        try:
            correct_idx = choice_labels.index(answer_key)
        except ValueError:
            errors += 1
            continue
        
        # Format question for scoring (match training format: just raw question)
        full_q = question
        
        # Score each choice
        best_score = float("-inf")
        best_idx = -1
        
        try:
            for i, (label, text) in enumerate(zip(choice_labels, choice_texts)):
                score = model._compute_choice_score(full_q, text, tokenizer, device)
                
                if score > best_score:
                    best_score = score
                    best_idx = i
        except Exception as e:
            errors += 1
            continue
        
        if best_idx == correct_idx:
            correct += 1
        total += 1
        
        if total % 100 == 0:
            print(f"  Progress: {total}/{len(arc_ds[split])} | Acc: {correct/total:.4f} | Errors: {errors}")
    
    accuracy = correct / total
    print(f"\nARC-Easy Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return {"arc_easy_accuracy": accuracy, "correct": correct, "total": total}


# ============================================================================
# Direction Quality Analysis
# ============================================================================

def analyze_directions(model, tokenizer, device, config):
    """
    Analyze the quality of learned reasoning directions.
    
    Tests:
    1. Within vs. across category similarity
    2. k-NN classification accuracy
    3. Direction variance and collapse detection
    """
    from data import UnifiedDataset
    
    dataset = UnifiedDataset(config, tokenizer, split="train")
    model.eval()
    
    all_directions = []
    all_coarse_ids = []
    all_fine_ids = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            ex = dataset[i]
            q = ex["question"]
            
            q_tok = tokenizer(q, return_tensors="pt", truncation=True, max_length=128)
            q_ids = q_tok["input_ids"].to(device)
            q_mask = q_tok["attention_mask"].to(device)
            
            q_outputs = model.base_model(input_ids=q_ids, attention_mask=q_mask, 
                                         output_hidden_states=True)
            q_hidden = q_outputs.hidden_states[-1]
            mask_exp = q_mask.unsqueeze(-1).float()
            h_pooled = (q_hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)
            
            d = model.direction_encoder(h_pooled)
            d_norm = F.normalize(d, dim=-1)
            
            all_directions.append(d_norm.cpu())
            all_coarse_ids.append(ex["coarse_category_id"])
            all_fine_ids.append(ex["fine_category_id"])
    
    directions = torch.cat(all_directions, dim=0)
    coarse_ids = torch.tensor(all_coarse_ids)
    fine_ids = torch.tensor(all_fine_ids)
    
    results = {}
    
    # Test 1: Within vs. across category similarity
    sim_matrix = directions @ directions.T
    
    for level, ids in [("coarse", coarse_ids), ("fine", fine_ids)]:
        same_mask = (ids.unsqueeze(0) == ids.unsqueeze(1)).float()
        same_mask.fill_diagonal_(0)
        diff_mask = 1 - same_mask
        diff_mask.fill_diagonal_(0)
        
        within_sim = (sim_matrix * same_mask).sum() / same_mask.sum().clamp(min=1)
        across_sim = (sim_matrix * diff_mask).sum() / diff_mask.sum().clamp(min=1)
        
        results[f"{level}_within_similarity"] = within_sim.item()
        results[f"{level}_across_similarity"] = across_sim.item()
        results[f"{level}_separation_ratio"] = (within_sim / across_sim.abs().clamp(min=0.01)).item()
    
    # Test 2: Direction variance
    results["direction_mean_norm"] = directions.norm(dim=-1).mean().item()
    results["direction_std_norm"] = directions.norm(dim=-1).std().item()
    results["direction_variance_per_dim"] = directions.var(dim=0).mean().item()
    
    # Test 3: k-NN classification
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        
        X = directions.numpy()
        
        for level, ids in [("coarse", coarse_ids), ("fine", fine_ids)]:
            y = ids.numpy()
            if len(set(y)) < 2:
                continue
            knn = KNeighborsClassifier(n_neighbors=min(5, len(y)//2))
            scores = cross_val_score(knn, X, y, cv=min(5, len(set(y))))
            results[f"{level}_knn_accuracy"] = scores.mean()
    except ImportError:
        print("  sklearn not available; skipping k-NN test")
    
    # Test 4: Collapse detection
    pairwise_sims = []
    for i in range(min(100, len(directions))):
        for j in range(i+1, min(100, len(directions))):
            pairwise_sims.append((directions[i] * directions[j]).sum().item())
    
    results["mean_pairwise_similarity"] = np.mean(pairwise_sims)
    results["std_pairwise_similarity"] = np.std(pairwise_sims)
    results["collapse_detected"] = bool(np.mean(pairwise_sims) > 0.9)
    
    # Print results
    print("\n" + "=" * 60)
    print("Direction Quality Analysis")
    print("=" * 60)
    for key, val in results.items():
        if isinstance(val, bool):
            print(f"  {key}: {'⚠️  YES' if val else '✅ No'}")
        elif isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    
    return results, directions, coarse_ids, fine_ids


def visualize_directions(directions, coarse_ids, fine_ids, 
                        output_path="analysis/directions_tsne.png"):
    """Create t-SNE visualization of directions."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("sklearn/matplotlib not available for visualization")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    X = directions.numpy() if isinstance(directions, torch.Tensor) else directions
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    X_2d = tsne.fit_transform(X)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    coarse_labels = ["Math", "Logic", "Language", "Commonsense", "Knowledge", "PatternRecognition"]
    colors = plt.cm.Set1(np.linspace(0, 1, len(coarse_labels)))
    
    for i, (label, color) in enumerate(zip(coarse_labels, colors)):
        mask = coarse_ids == i
        if mask.sum() > 0:
            ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], label=label, alpha=0.7, s=30)
    ax1.set_title("Directions by Coarse Category")
    ax1.legend(fontsize=8)
    
    unique_fine = sorted(set(fine_ids.tolist()))
    colors2 = plt.cm.Set2(np.linspace(0, 1, len(unique_fine)))
    for i, fid in enumerate(unique_fine):
        mask = fine_ids == fid
        if mask.sum() > 0:
            ax2.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors2[i]], alpha=0.7, s=30)
    ax2.set_title("Directions by Fine Category")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"t-SNE visualization saved to {output_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lrd/checkpoint_best")
    parser.add_argument("--eval_arc", action="store_true")
    parser.add_argument("--analyze_directions", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    args = parser.parse_args()
    
    model, tokenizer, device = load_checkpoint(args.checkpoint)
    config = model.config
    
    if args.analyze_directions:
        results, directions, coarse_ids, fine_ids = analyze_directions(model, tokenizer, device, config)
        
        os.makedirs("analysis", exist_ok=True)
        with open("analysis/direction_quality.json", "w") as f:
            json.dump({k: float(v) if not isinstance(v, (np.floating, np.bool_)) else v 
                       for k, v in results.items()}, f, indent=2)
        
        if args.visualize:
            visualize_directions(directions, coarse_ids, fine_ids)
    
    if args.eval_arc:
        results = evaluate_arc_easy(model, tokenizer, device, config, args.num_fewshot)
        
        os.makedirs("results", exist_ok=True)
        with open("results/arc_easy_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to results/arc_easy_results.json")
