"""
Tokenizer utilities for Deep Reasoning LLM.
Uses SentencePiece with Unigram algorithm for optimal subword tokenization.
"""

import os
from typing import List, Optional
import sentencepiece as spm
from pathlib import Path


class TokenizerTrainer:
    """
    Trainer for SentencePiece Unigram tokenizer.
    Optimized for reasoning and general language tasks.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        character_coverage: float = 0.9995,
        model_type: str = "unigram",
    ):
        """
        Args:
            vocab_size: Target vocabulary size
            character_coverage: Character coverage for the tokenizer
            model_type: SentencePiece model type (unigram, bpe, char, word)
        """
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage
        self.model_type = model_type
        
    def train(
        self,
        input_files: List[str],
        model_prefix: str,
        special_tokens: Optional[List[str]] = None,
        num_threads: int = 16,
    ):
        """
        Train SentencePiece tokenizer on input data.
        
        Args:
            input_files: List of paths to training text files
            model_prefix: Prefix for output model files
            special_tokens: Additional special tokens
            num_threads: Number of threads for training
        """
        if special_tokens is None:
            special_tokens = [
                "<pad>",     # Padding token
                "<unk>",     # Unknown token
                "<s>",       # Beginning of sequence
                "</s>",      # End of sequence
                "<think>",   # Start reasoning
                "</think>",  # End reasoning
                "<step>",    # Reasoning step
                "<conclusion>",  # Final answer
            ]
        
        # Ensure output directory exists
        output_dir = Path(model_prefix).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # SentencePiece handles <pad>, <unk>, <s>, </s> automatically via special IDs
        # Only include custom reasoning tokens in user_defined_symbols
        custom_tokens = ["<think>", "</think>", "<step>", "<conclusion>"]
        
        # Prepare training arguments
        train_args = {
            "input": ",".join(input_files),
            "model_prefix": model_prefix,
            "vocab_size": self.vocab_size,
            "character_coverage": self.character_coverage,
            "model_type": self.model_type,
            "user_defined_symbols": ",".join(custom_tokens),
            "pad_id": 0,
            "unk_id": 1,
            "bos_id": 2,
            "eos_id": 3,
            "pad_piece": "<pad>",
            "unk_piece": "<unk>",
            "bos_piece": "<s>",
            "eos_piece": "</s>",
            "num_threads": num_threads,
            "train_extremely_large_corpus": False,
            # Additional parameters for better quality
            "split_by_unicode_script": True,
            "split_by_whitespace": True,
            "split_by_number": True,
            "max_sentence_length": 16384,
            "shuffle_input_sentence": True,
            "seed_sentencepiece_size": 1000000,
            # Normalization
            "normalization_rule_name": "nmt_nfkc_cf",
            "remove_extra_whitespaces": True,
            "max_sentencepiece_length": 16,
            "byte_fallback": True,  # Handle any byte sequence
        }
        
        # Convert to command line arguments
        train_args_str = " ".join([f"--{k}={v}" for k, v in train_args.items()])
        
        print("=" * 60)
        print("Training SentencePiece Tokenizer")
        print("=" * 60)
        print(f"Model Type: {self.model_type}")
        print(f"Vocabulary Size: {self.vocab_size:,}")
        print(f"Character Coverage: {self.character_coverage}")
        print(f"Input Files: {len(input_files)}")
        print(f"Special Tokens: {len(special_tokens)}")
        print("=" * 60)
        
        # Train the model
        spm.SentencePieceTrainer.Train(train_args_str)
        
        print(f"✓ Tokenizer saved to: {model_prefix}.model")
        print(f"✓ Vocabulary saved to: {model_prefix}.vocab")
        
        return f"{model_prefix}.model", f"{model_prefix}.vocab"
    
    @staticmethod
    def test_tokenizer(model_path: str, test_texts: Optional[List[str]] = None):
        """
        Test the trained tokenizer on sample texts.
        
        Args:
            model_path: Path to the trained .model file
            test_texts: Optional list of test texts
        """
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        
        if test_texts is None:
            test_texts = [
                "Let's think step by step to solve this problem.",
                "The quick brown fox jumps over the lazy dog.",
                "To solve 2 + 2, I need to add the numbers: 2 + 2 = 4",
                "Chain-of-thought reasoning helps in complex problem solving.",
            ]
        
        print("\n" + "=" * 60)
        print("Tokenizer Test")
        print("=" * 60)
        print(f"Vocabulary Size: {sp.vocab_size():,}")
        print(f"BOS ID: {sp.bos_id()} ({sp.id_to_piece(sp.bos_id())})")
        print(f"EOS ID: {sp.eos_id()} ({sp.id_to_piece(sp.eos_id())})")
        print(f"PAD ID: {sp.pad_id()} ({sp.id_to_piece(sp.pad_id())})")
        print(f"UNK ID: {sp.unk_id()} ({sp.id_to_piece(sp.unk_id())})")
        print("=" * 60)
        
        for text in test_texts:
            tokens = sp.encode_as_pieces(text)
            ids = sp.encode_as_ids(text)
            print(f"\nText: {text}")
            print(f"Tokens ({len(tokens)}): {tokens}")
            print(f"IDs: {ids}")
            print(f"Decoded: {sp.decode_pieces(tokens)}")
        
        return sp


def prepare_training_data(input_dir: str, output_file: str, max_files: Optional[int] = None):
    """
    Prepare training data by combining multiple text files.
    
    Args:
        input_dir: Directory containing raw text files
        output_file: Output file path for combined data
        max_files: Maximum number of files to process
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all text files
    text_files = list(input_path.glob("**/*.txt"))
    if max_files:
        text_files = text_files[:max_files]
    
    print(f"Found {len(text_files)} text files")
    
    total_lines = 0
    with open(output_path, 'w', encoding='utf-8') as outf:
        for i, file_path in enumerate(text_files):
            print(f"Processing {i+1}/{len(text_files)}: {file_path.name}")
            with open(file_path, 'r', encoding='utf-8') as inf:
                lines = inf.readlines()
                outf.writelines(lines)
                total_lines += len(lines)
    
    print(f"\n✓ Combined {total_lines:,} lines into {output_path}")
    return str(output_path)


if __name__ == "__main__":
    # Example usage
    print("SentencePiece Tokenizer Trainer")
    print("This script will be called by scripts/train_tokenizer.py")
