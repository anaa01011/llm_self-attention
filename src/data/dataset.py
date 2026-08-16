"""
Data processing and loading utilities for Deep Reasoning LLM.
Handles various data formats and creates efficient dataloaders.
"""

import os
import json
import random
from typing import Dict, List, Optional, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
import datasets
from datasets import load_dataset


class ReasoningDataset(Dataset):
    """
    Dataset for reasoning-focused language modeling.
    Supports chain-of-thought and general text data.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        data_type: str = "jsonl",
        include_reasoning: bool = True,
    ):
        """
        Args:
            data_path: Path to data file or directory
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            data_type: Type of data file (jsonl, txt, parquet)
            include_reasoning: Whether to add reasoning tokens
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_reasoning = include_reasoning
        
        # Load data
        self.examples = self._load_data(data_path, data_type)
        print(f"Loaded {len(self.examples)} examples from {data_path}")
    
    def _load_data(self, data_path: str, data_type: str) -> List[Dict]:
        """Load data from file."""
        examples = []
        path = Path(data_path)
        
        if data_type == "jsonl":
            # Load JSONL format (one JSON per line)
            if path.is_file():
                files = [path]
            else:
                files = list(path.glob("**/*.jsonl"))
            
            for file in files:
                with open(file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            examples.append(json.loads(line))
        
        elif data_type == "txt":
            # Load plain text files
            if path.is_file():
                files = [path]
            else:
                files = list(path.glob("**/*.txt"))
            
            for file in files:
                with open(file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    # Split into chunks
                    chunks = text.split('\n\n')
                    for chunk in chunks:
                        if len(chunk.strip()) > 50:
                            examples.append({"text": chunk.strip()})
        
        elif data_type == "hf":
            # Load from HuggingFace datasets
            dataset = load_dataset(data_path)
            examples = [item for item in dataset['train']]
        
        return examples
    
    def _format_reasoning_example(self, example: Dict) -> str:
        """
        Format example with reasoning tokens.
        Supports chain-of-thought format.
        """
        if "reasoning" in example and "answer" in example:
            # Chain-of-thought format
            text = f"{example.get('question', '')}\n"
            text += f"<think>{example['reasoning']}</think>\n"
            text += f"<conclusion>{example['answer']}</conclusion>"
            return text
        elif "text" in example:
            return example["text"]
        elif "input" in example and "output" in example:
            return f"{example['input']}\n{example['output']}"
        else:
            # Fallback: use first text field
            for key in ["content", "prompt", "instruction"]:
                if key in example:
                    return example[key]
        
        return str(example)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        example = self.examples[idx]
        
        # Format the text
        text = self._format_reasoning_example(example)
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Prepare labels (for causal LM, labels = input_ids)
        labels = encoded["input_ids"].clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


class MixedDataset(Dataset):
    """
    Dataset that mixes multiple data sources with specified weights.
    Useful for combining reasoning, code, and general text data.
    """
    
    def __init__(
        self,
        datasets: List[Dataset],
        weights: Optional[List[float]] = None,
        size: Optional[int] = None,
    ):
        """
        Args:
            datasets: List of datasets to mix
            weights: Sampling weights for each dataset (must sum to 1)
            size: Total size of mixed dataset (default: sum of all datasets)
        """
        self.datasets = datasets
        
        if weights is None:
            weights = [1.0 / len(datasets)] * len(datasets)
        
        assert len(weights) == len(datasets), "Number of weights must match number of datasets"
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
        
        self.weights = weights
        self.size = size or sum(len(d) for d in datasets)
        
        # Precompute sampling indices
        self.indices = self._create_sampling_indices()
    
    def _create_sampling_indices(self) -> List[tuple]:
        """Create list of (dataset_idx, example_idx) tuples."""
        indices = []
        for _ in range(self.size):
            # Sample dataset according to weights
            dataset_idx = random.choices(range(len(self.datasets)), weights=self.weights)[0]
            # Sample example from that dataset
            example_idx = random.randint(0, len(self.datasets[dataset_idx]) - 1)
            indices.append((dataset_idx, example_idx))
        return indices
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_idx, example_idx = self.indices[idx]
        return self.datasets[dataset_idx][example_idx]


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with optimal settings.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    # pin_memory is only supported on CUDA; disable for MPS and CPU
    import torch
    effective_pin_memory = pin_memory and torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        drop_last=True,  # Drop last incomplete batch
    )


def prepare_datasets(
    train_path: str,
    eval_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    data_type: str = "jsonl",
) -> tuple:
    """
    Prepare training and evaluation datasets.
    
    Args:
        train_path: Path to training data
        eval_path: Path to evaluation data
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        data_type: Type of data file
        
    Returns:
        (train_dataset, eval_dataset)
    """
    train_dataset = ReasoningDataset(
        train_path,
        tokenizer,
        max_length=max_length,
        data_type=data_type,
    )
    
    eval_dataset = ReasoningDataset(
        eval_path,
        tokenizer,
        max_length=max_length,
        data_type=data_type,
    )
    
    return train_dataset, eval_dataset


if __name__ == "__main__":
    print("Data processing utilities")
    print("Use this module via training scripts")
