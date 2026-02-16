#!/usr/bin/env python3
"""
Script to train SentencePiece tokenizer on custom data.

Usage:
    python scripts/train_tokenizer.py --data_dir data/raw --output_dir tokenizer/
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizer import TokenizerTrainer, prepare_training_data


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing training text files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tokenizer",
        help="Output directory for tokenizer files"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50000,
        help="Vocabulary size"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="unigram",
        choices=["unigram", "bpe", "char", "word"],
        help="SentencePiece model type"
    )
    parser.add_argument(
        "--character_coverage",
        type=float,
        default=0.9995,
        help="Character coverage"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=16,
        help="Number of threads for training"
    )
    parser.add_argument(
        "--prepare_data",
        action="store_true",
        help="Combine multiple files into single training file"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare training data if requested
    if args.prepare_data:
        print("Preparing training data...")
        combined_file = prepare_training_data(
            args.data_dir,
            output_dir / "combined_train.txt"
        )
        input_files = [combined_file]
    else:
        # Find all text files in data directory
        data_path = Path(args.data_dir)
        input_files = list(data_path.glob("**/*.txt"))
        
        if not input_files:
            print(f"Error: No .txt files found in {args.data_dir}")
            print("Please provide training data or use --prepare_data flag")
            sys.exit(1)
        
        input_files = [str(f) for f in input_files]
    
    print(f"\nFound {len(input_files)} training file(s)")
    
    # Initialize trainer
    trainer = TokenizerTrainer(
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
    )
    
    # Train tokenizer
    model_prefix = str(output_dir / "tokenizer")
    
    print("\nStarting training...")
    model_file, vocab_file = trainer.train(
        input_files=input_files,
        model_prefix=model_prefix,
        num_threads=args.num_threads,
    )
    
    # Test the tokenizer
    print("\nTesting tokenizer...")
    TokenizerTrainer.test_tokenizer(model_file)
    
    print("\n" + "=" * 60)
    print("Tokenizer Training Complete!")
    print("=" * 60)
    print(f"Model file: {model_file}")
    print(f"Vocab file: {vocab_file}")
    print("\nYou can now use this tokenizer for model training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
