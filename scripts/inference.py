#!/usr/bin/env python3
"""
Inference script for Deep Reasoning LLM.
Generate text using trained model.

Usage:
    python scripts/inference.py --checkpoint checkpoints/checkpoint-10000 --prompt "Solve: 2+2="
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model import ReasoningLLMForCausalLM
from src.tokenizer import load_tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 200,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
):
    """
    Generate text using the model.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p (nucleus) sampling
        num_return_sequences: Number of sequences to generate
        
    Returns:
        List of generated texts
    """
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )
    
    # Decode
    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts


def main():
    parser = argparse.ArgumentParser(description="Generate text with Deep Reasoning LLM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenizer/tokenizer.model",
        help="Path to tokenizer model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Let's think step by step:",
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=1,
        help="Number of sequences to generate"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode"
    )
    
    args = parser.parse_args()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer_path)
    print(f"✓ Tokenizer loaded")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = ReasoningLLMForCausalLM.from_pretrained(args.checkpoint)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    if args.interactive:
        # Interactive mode
        print("\n" + "=" * 60)
        print("Interactive Generation Mode")
        print("Type 'quit' or 'exit' to stop")
        print("=" * 60 + "\n")
        
        while True:
            try:
                prompt = input("\nPrompt: ")
                
                if prompt.lower() in ['quit', 'exit']:
                    break
                
                if not prompt.strip():
                    continue
                
                print("\nGenerating...")
                generated_texts = generate_text(
                    model,
                    tokenizer,
                    prompt,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    num_return_sequences=args.num_sequences,
                )
                
                print("\n" + "-" * 60)
                for i, text in enumerate(generated_texts, 1):
                    print(f"\nGeneration {i}:")
                    print(text)
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
    else:
        # Single generation
        print(f"\nPrompt: {args.prompt}")
        print("\nGenerating...")
        
        generated_texts = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_sequences,
        )
        
        print("\n" + "=" * 60)
        print("Generated Text")
        print("=" * 60)
        
        for i, text in enumerate(generated_texts, 1):
            print(f"\nGeneration {i}:")
            print(text)
            print("-" * 60)


if __name__ == "__main__":
    main()
