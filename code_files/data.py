"""
Latent Reasoning Direction (LRD) - Data Preprocessing Pipeline

Handles all three bench-labs datasets with different schemas and normalizes
them into a unified format with coarse/fine category labels.
"""
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, List, Tuple
from config import LRDConfig


class UnifiedDataset(Dataset):
    """
    Unified dataset that normalizes all three bench-labs datasets.
    
    Each record contains:
    - question: str
    - answer: str
    - coarse_category: str (one of 6 top-level)
    - fine_category: str (one of ~17 subcategories)
    - coarse_category_id: int
    - fine_category_id: int
    """
    
    def __init__(self, config: LRDConfig, tokenizer, split: str = "train", val_fraction: float = 0.1):
        self.config = config
        self.tokenizer = tokenizer
        self.examples = []
        
        # Build category ID mappings
        self.coarse_to_id = {cat: i for i, cat in enumerate(config.coarse_categories)}
        
        # Collect all fine categories across datasets
        all_fine_cats = set()
        raw_data = self._load_raw_datasets()
        for record in raw_data:
            all_fine_cats.add(record["fine_category"])
        self.fine_categories = sorted(all_fine_cats)
        self.fine_to_id = {cat: i for i, cat in enumerate(self.fine_categories)}
        
        # Split into train/val
        import random
        random.seed(42)
        random.shuffle(raw_data)
        val_size = max(1, int(len(raw_data) * val_fraction))
        
        if split == "val":
            raw_data = raw_data[:val_size]
        else:
            raw_data = raw_data[val_size:]
        
        self.examples = raw_data
    
    def _load_raw_datasets(self) -> List[Dict]:
        """Load and normalize all three datasets."""
        all_records = []
        
        for ds_name in self.config.dataset_names:
            ds = load_dataset(ds_name)
            split_name = list(ds.keys())[0]
            
            source = self._get_source_name(ds_name)
            
            for example in ds[split_name]:
                record = self._normalize_record(example, source)
                all_records.append(record)
        
        return all_records
    
    def _get_source_name(self, ds_name: str) -> str:
        if "effortless" in ds_name:
            return "effortless"
        elif "easy" in ds_name:
            return "easy"
        elif "mid" in ds_name:
            return "mid"
        return "unknown"
    
    def _normalize_record(self, record: Dict, source: str) -> Dict:
        """Normalize a single record to unified schema."""
        cfg = self.config
        
        if source == "effortless":
            question = record["question"]
            answer = record["answer"]
            coarse_cat = record["category"]
            fine_cat = coarse_cat  # No finer granularity available
            
        elif source == "easy":
            question = record["question"]
            answer = record["answer"]
            fine_cat = record["category"]
            coarse_cat = cfg.fine_to_coarse.get(fine_cat, fine_cat.split("-")[0])
            
        elif source == "mid":
            question = record["input"]
            # Extract correct answer from target_scores
            target_scores = record["target_scores"]
            correct_answers = [k for k, v in target_scores.items() if v == 1.0]
            answer = correct_answers[0] if correct_answers else list(target_scores.keys())[0]
            
            fine_cat = record["category"]
            coarse_cat = cfg.fine_to_coarse.get(fine_cat, fine_cat.split("-")[0])
            
            # Format as multiple choice in the question
            choices = list(target_scores.keys())
            if len(choices) > 1:
                choice_text = " ".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
                question = f"{question}\nChoices: {choice_text}"
        else:
            raise ValueError(f"Unknown source: {source}")
        
        return {
            "question": question,
            "answer": answer,
            "coarse_category": coarse_cat,
            "fine_category": fine_cat,
            "coarse_category_id": self.coarse_to_id.get(coarse_cat, 0),
            "fine_category_id": -1,  # Will be set after all categories are collected
            "source": source,
        }
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Set fine_category_id (needs to be done after __init__ collects all fine cats)
        fine_id = self.fine_to_id.get(ex["fine_category"], 0)
        
        return {
            "question": ex["question"],
            "answer": ex["answer"],
            "coarse_category_id": ex["coarse_category_id"],
            "fine_category_id": fine_id,
            "coarse_category": ex["coarse_category"],
            "fine_category": ex["fine_category"],
        }


def collate_fn(batch, tokenizer, max_seq_len=256):
    """
    Custom collate function that tokenizes questions and answers separately,
    then concatenates them with proper loss masks.
    
    Returns:
    - input_ids: [B, L] concatenated token IDs
    - attention_mask: [B, L]
    - loss_mask: [B, L] with 1.0 only at answer token positions
    - prefix_mask: [B, L] with 1.0 only at prefix position(s)
    - coarse_category_ids: [B]
    - fine_category_ids: [B]
    - question_lens: [B] lengths of question portions
    """
    B = len(batch)
    
    questions = [ex["question"] for ex in batch]
    answers = [ex["answer"] for ex in batch]
    coarse_ids = torch.tensor([ex["coarse_category_id"] for ex in batch])
    fine_ids = torch.tensor([ex["fine_category_id"] for ex in batch])
    
    # Tokenize questions and answers separately
    q_tok = tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=max_seq_len // 2,
        return_tensors="pt",
    )
    a_tok = tokenizer(
        answers,
        padding=True,
        truncation=True,
        max_length=max_seq_len // 2,
        return_tensors="pt",
    )
    
    # Concatenate question and answer tokens
    q_ids = q_tok["input_ids"]       # [B, L_q]
    q_mask = q_tok["attention_mask"]  # [B, L_q]
    a_ids = a_tok["input_ids"]       # [B, L_a]
    a_mask = a_tok["attention_mask"]  # [B, L_a]
    
    # We'll add 1 position for the direction prefix
    num_prefix = 1
    L_q = q_ids.shape[1]
    L_a = a_ids.shape[1]
    L_total = num_prefix + L_q + L_a
    
    # Build full sequences
    input_ids = torch.zeros(B, L_total, dtype=torch.long)
    attention_mask = torch.zeros(B, L_total, dtype=torch.long)
    loss_mask = torch.zeros(B, L_total, dtype=torch.float)
    
    # Prefix position (token ID doesn't matter; we'll replace with direction embedding)
    prefix_token_id = tokenizer.eos_token_id
    input_ids[:, 0] = prefix_token_id
    attention_mask[:, 0] = 1
    
    # Question tokens
    input_ids[:, num_prefix:num_prefix + L_q] = q_ids
    attention_mask[:, num_prefix:num_prefix + L_q] = q_mask
    
    # Answer tokens
    input_ids[:, num_prefix + L_q:] = a_ids
    attention_mask[:, num_prefix + L_q:] = a_mask
    
    # Loss mask: only answer tokens (and only where attention_mask is 1)
    loss_mask[:, num_prefix + L_q:] = a_mask.float()
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "coarse_category_ids": coarse_ids,
        "fine_category_ids": fine_ids,
        "question_lens": q_mask.sum(dim=1),  # Actual question lengths
        "num_prefix": num_prefix,
    }


def create_dataloaders(config: LRDConfig, tokenizer):
    """Create train and validation dataloaders."""
    from torch.utils.data import DataLoader
    from functools import partial
    
    train_dataset = UnifiedDataset(config, tokenizer, split="train")
    val_dataset = UnifiedDataset(config, tokenizer, split="val")
    
    collate = partial(collate_fn, tokenizer=tokenizer, max_seq_len=config.max_seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    
    return train_loader, val_loader, train_dataset
