"""
Diagnostic script to debug why ARC-Easy evaluation gives 0% accuracy.

Checks:
1. Checkpoint exists and loads correctly
2. Model is on correct device
3. Direction encoder works
4. Scoring works on training examples
"""
import os
import torch
from train import load_checkpoint
from data import UnifiedDataset
from config import LRDConfig
from transformers import AutoTokenizer

def diagnose(checkpoint_path="checkpoints/lrd/checkpoint_best"):
    print("=" * 70)
    print("LRD MODEL DIAGNOSTIC")
    print("=" * 70)
    
    # 1. Check checkpoint exists
    print("\n1. Checking checkpoint...")
    if not os.path.exists(checkpoint_path):
        print(f"   ❌ Checkpoint not found: {checkpoint_path}")
        print(f"   Available checkpoints:")
        if os.path.exists("checkpoints/lrd"):
            for item in os.listdir("checkpoints/lrd"):
                print(f"     - {item}")
        return
    
    print(f"   ✅ Checkpoint exists: {checkpoint_path}")
    
    # Check files
    required_files = ["config.json", "model.safetensors", "direction_modules.pt"]
    for f in required_files:
        path = os.path.join(checkpoint_path, f)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024*1024)
            print(f"   ✅ {f} ({size:.1f} MB)")
        else:
            print(f"   ❌ {f} missing")
    
    # 2. Load model
    print("\n2. Loading model...")
    try:
        model, tokenizer, device = load_checkpoint(checkpoint_path)
        print(f"   ✅ Model loaded on {device}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        return
    
    # 3. Check tokenizer
    print("\n3. Testing tokenizer...")
    test_q = "What is 2 + 2?"
    print(f"   Input: '{test_q}'")
    
    try:
        q_tok = tokenizer(test_q, return_tensors="pt", truncation=True, max_length=128)
        q_ids = q_tok["input_ids"]
        q_mask = q_tok["attention_mask"]
        print(f"   Token IDs shape: {q_ids.shape}")
        print(f"   Token IDs: {q_ids[0].tolist()[:10]}...")
        print(f"   Decoded: '{tokenizer.decode(q_ids[0])}'")
        
        if q_ids.shape[1] == 0:
            print(f"   ❌ Tokenizer produced 0 tokens!")
            print(f"   Tokenizer vocab size: {tokenizer.vocab_size}")
            print(f"   Tokenizer pad_token: {tokenizer.pad_token}")
            print(f"   Tokenizer eos_token: {tokenizer.eos_token}")
            return
        else:
            print(f"   ✅ Tokenizer works")
    except Exception as e:
        print(f"   ❌ Tokenizer failed: {e}")
        return
    
    # 4. Check direction encoder
    print("\n4. Testing direction encoder...")
    model.eval()
    q_ids = q_ids.to(device)
    q_mask = q_mask.to(device)
    
    with torch.no_grad():
        q_out = model.base_model(input_ids=q_ids, attention_mask=q_mask, output_hidden_states=True)
        q_h = q_out.hidden_states[-1]
        me = q_mask.unsqueeze(-1).float()
        hp = (q_h * me).sum(1) / me.sum(1).clamp(min=1)
        d = model.direction_encoder(hp)
        d_norm = torch.nn.functional.normalize(d, dim=-1)
    
    print(f"   Direction shape: {d_norm.shape}")
    print(f"   Direction norm: {d_norm.norm().item():.4f}")
    print(f"   Direction range: [{d_norm.min().item():.4f}, {d_norm.max().item():.4f}]")
    print(f"   ✅ Direction encoder works")
    
    # 5. Test on training example
    print("\n5. Testing on training example...")
    config = LRDConfig()
    train_dataset = UnifiedDataset(config, tokenizer, split="train")
    
    if len(train_dataset) > 0:
        ex = train_dataset[0]
        question = ex["question"]
        answer = ex["answer"]
        print(f"   Question: {question[:80]}...")
        print(f"   Answer: {answer}")
        
        # Score the correct answer
        score_correct = model._compute_choice_score(question, answer, tokenizer, device)
        print(f"   Score (correct answer): {score_correct:.4f}")
        
        # Score a wrong answer
        wrong_answer = "This is definitely wrong"
        score_wrong = model._compute_choice_score(question, wrong_answer, tokenizer, device)
        print(f"   Score (wrong answer): {score_wrong:.4f}")
        
        if score_correct > score_wrong:
            print(f"   ✅ Model prefers correct answer (diff: {score_correct - score_wrong:.4f})")
        else:
            print(f"   ⚠️  Model prefers wrong answer (diff: {score_wrong - score_correct:.4f})")
            print(f"   This suggests the model didn't learn much from training")
    
    # 6. Test generation
    print("\n6. Testing generation...")
    try:
        output = model.generate_with_direction("What is 2 + 2?", tokenizer, max_new_tokens=10)
        print(f"   Generated: {output[:100]}")
        print(f"   ✅ Generation works")
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
    
    # 7. Test ARC-Easy style scoring
    print("\n7. Testing ARC-Easy style scoring...")
    arc_q = "What will happen to an ice cube left on the kitchen counter at room temperature?"
    arc_choices = ["It will melt", "It will freeze", "It will evaporate instantly", "It will stay the same"]
    
    scores = []
    for w in arc_choices:
        s = model._compute_choice_score(arc_q, w, tokenizer, device)
        scores.append((w, s))
        print(f"   '{w}': {s:.4f}")
    
    scores.sort(key=lambda x: x[1], reverse=True)
    print(f"   Best: '{scores[0][0]}' (score: {scores[0][1]:.4f})")
    print(f"   Correct: 'It will melt'")
    
    if scores[0][0] == "It will melt":
        print(f"   ✅ Model picks correct answer")
    else:
        print(f"   ⚠️  Model picks wrong answer")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lrd/checkpoint_best")
    args = parser.parse_args()
    
    diagnose(args.checkpoint)
