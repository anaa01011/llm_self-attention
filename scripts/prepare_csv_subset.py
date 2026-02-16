#!/usr/bin/env python3
"""
Extract a subset from train.csv for initial training validation.
Converts CSV format to reasoning-optimized text format.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import json
import random
from pathlib import Path


def format_example(row):
    """Convert CSV row to reasoning format with special tokens."""
    instruction = row.get('instruction', '').strip()
    output = row.get('output', '').strip()
    input_text = row.get('input', '').strip()
    
    # Skip empty examples
    if not instruction or not output:
        return None
    
    # Format with reasoning tokens
    if input_text:
        text = f"Question: {instruction}\nContext: {input_text}\n\n"
    else:
        text = f"Question: {instruction}\n\n"
    
    text += f"<think>{output}</think>"
    
    return {"text": text, "type": "reasoning", "source": row.get('data_source', 'unknown')}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract subset from train.csv')
    parser.add_argument('--input', default='train.csv', help='Input CSV file')
    parser.add_argument('--output-dir', default='data', help='Output directory')
    parser.add_argument('--num-examples', type=int, default=2000, help='Number of examples to extract')
    parser.add_argument('--train-ratio', type=float, default=0.9, help='Train/eval split ratio')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for sampling')
    args = parser.parse_args()
    
    print("=" * 60)
    print("CSV Dataset Subset Extractor")
    print("=" * 60)
    
    # Set random seed
    random.seed(args.random_seed)
    
    # Read CSV and extract examples
    print(f"\nReading {args.input}...")
    examples = []
    total_rows = 0
    skipped = 0
    
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            formatted = format_example(row)
            if formatted:
                examples.append(formatted)
            else:
                skipped += 1
            
            # Progress indicator
            if total_rows % 10000 == 0:
                print(f"  Processed {total_rows:,} rows, extracted {len(examples):,} examples...")
    
    print(f"\nTotal rows: {total_rows:,}")
    print(f"Valid examples: {len(examples):,}")
    print(f"Skipped (empty): {skipped:,}")
    
    # Sample subset
    if len(examples) > args.num_examples:
        print(f"\nSampling {args.num_examples:,} examples from {len(examples):,}...")
        sampled = random.sample(examples, args.num_examples)
    else:
        print(f"\nUsing all {len(examples):,} examples...")
        sampled = examples
    
    # Split train/eval
    random.shuffle(sampled)
    split_idx = int(len(sampled) * args.train_ratio)
    train_examples = sampled[:split_idx]
    eval_examples = sampled[split_idx:]
    
    print(f"\nSplit:")
    print(f"  Training: {len(train_examples):,} examples")
    print(f"  Evaluation: {len(eval_examples):,} examples")
    
    # Create directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{args.output_dir}/raw").mkdir(parents=True, exist_ok=True)
    Path(f"{args.output_dir}/train").mkdir(parents=True, exist_ok=True)
    Path(f"{args.output_dir}/eval").mkdir(parents=True, exist_ok=True)
    
    # Save raw text for tokenizer
    print("\nSaving raw text for tokenizer...")
    with open(f"{args.output_dir}/raw/train_text.txt", "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(ex["text"] + "\n\n")
    
    # Save JSONL for training
    print("Saving JSONL for training...")
    with open(f"{args.output_dir}/train/train.jsonl", "w", encoding="utf-8") as f:
        for ex in train_examples:
            json.dump(ex, f)
            f.write("\n")
    
    with open(f"{args.output_dir}/eval/eval.jsonl", "w", encoding="utf-8") as f:
        for ex in eval_examples:
            json.dump(ex, f)
            f.write("\n")
    
    # Show sample
    print("\n" + "=" * 60)
    print("Sample Example:")
    print("=" * 60)
    print(train_examples[0]["text"][:500])
    if len(train_examples[0]["text"]) > 500:
        print("...(truncated)")
    
    print("\n" + "=" * 60)
    print("✓ Dataset subset prepared!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  {args.output_dir}/raw/train_text.txt - Raw text for tokenizer")
    print(f"  {args.output_dir}/train/train.jsonl - Training data")
    print(f"  {args.output_dir}/eval/eval.jsonl - Evaluation data")
    print(f"\nNext steps:")
    print(f"  1. Train tokenizer: python scripts/train_tokenizer.py")
    print(f"  2. Start training: python scripts/train.py --config configs/base_config.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()
