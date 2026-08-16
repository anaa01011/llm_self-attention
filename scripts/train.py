#!/usr/bin/env python3
"""
Main training script for Deep Reasoning LLM.

Usage:
    python scripts/train.py --config configs/base_config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file if present (for HF_TOKEN etc.)
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

import torch
from src.model import ReasoningLLMConfig, ReasoningLLMForCausalLM
from src.tokenizer import load_tokenizer
from src.data import prepare_datasets, create_dataloader
from src.training import Trainer
from src.utils import set_seed, load_config, model_summary


def main():
    parser = argparse.ArgumentParser(description="Train Deep Reasoning LLM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenizer/tokenizer.model",
        help="Path to trained tokenizer model"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config["checkpointing"]["output_dir"] = args.output_dir
    
    # Set random seed for reproducibility
    set_seed(config.get("seed", 42))
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer_path)
    print(f" Tokenizer loaded with vocab size: {tokenizer.vocab_size}")
    
    # Create model configuration
    print("\nCreating model configuration...")
    model_config = ReasoningLLMConfig(
        vocab_size=config["model"]["vocab_size"],
        hidden_size=config["model"]["hidden_size"],
        num_hidden_layers=config["model"]["num_hidden_layers"],
        num_attention_heads=config["model"]["num_attention_heads"],
        intermediate_size=config["model"]["intermediate_size"],
        max_position_embeddings=config["model"]["max_position_embeddings"],
        hidden_dropout_prob=config["model"]["hidden_dropout_prob"],
        attention_probs_dropout_prob=config["model"]["attention_probs_dropout_prob"],
        layer_norm_eps=config["model"]["layer_norm_eps"],
        hidden_act=config["model"]["hidden_act"],
        initializer_range=config["model"]["initializer_range"],
        enable_reasoning_tokens=config["model"]["enable_reasoning_tokens"],
        reasoning_depth=config["model"]["reasoning_depth"],
        use_cache=config["model"]["use_cache"],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    model_config.print_config()
    
    # Create model
    print("\nInitializing model...")
    if args.resume_from:
        print(f"Loading model from checkpoint: {args.resume_from}")
        model = ReasoningLLMForCausalLM.from_pretrained(args.resume_from)
    else:
        model = ReasoningLLMForCausalLM(model_config)
    
    # Print model summary
    model_summary(model)
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset, eval_dataset = prepare_datasets(
        train_path=config["data"]["train_data_path"],
        eval_path=config["data"]["eval_data_path"],
        tokenizer=tokenizer,
        max_length=config["data"]["max_seq_length"],
        data_type="jsonl",  # Default to JSONL
    )
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
    )
    
    eval_dataloader = create_dataloader(
        eval_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )
    
    print(f" Training samples: {len(train_dataset):,}")
    print(f" Evaluation samples: {len(eval_dataset):,}")
    
    # Create training configuration
    training_config = {
        **config["training"],
        **config["checkpointing"],
        **config["logging"],
    }
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=training_config,
    )
    
    # Start training
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint(trainer.global_step)
        print(" Checkpoint saved")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
