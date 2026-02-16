#!/usr/bin/env python3
"""
Quick inference test with the trained checkpoint
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model.model import ReasoningLLMForCausalLM
from src.tokenizer.tokenizer import ReasoningLLMTokenizer


def test_inference(checkpoint_path="checkpoints/test_run/checkpoint-200"):
    print("=" * 60)
    print("Testing Trained Model Inference")
    print("=" * 60)
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = ReasoningLLMTokenizer("tokenizer/tokenizer.model")
    print(f"   ✓ Vocab size: {len(tokenizer):,}")
    
    # Load model
    print("\n2. Loading trained model...")
    model = ReasoningLLMForCausalLM.from_pretrained(checkpoint_path)
    model.eval()
    print(f"   ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   ✓ Device: {next(model.parameters()).device}")
    
    # Test prompts
    test_prompts = [
        "Question: What is 5 + 3?\n\n",
        "Question: If a train travels 60 miles in 2 hours, what is its speed?\n\n",
        "Question: What is 2 * 4?\n\n",
    ]
    
    print("\n" + "=" * 60)
    print("Generating Responses")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Test {i}]")
        print(f"Prompt: {prompt.strip()}")
        print("-" * 60)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,  # Disable KV cache for simplicity
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"Generated: {response}")
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("✓ Inference test complete!")
    print("=" * 60)
    print("\nNote: This is a very early checkpoint (200 steps).")
    print("Expect basic token patterns, not coherent reasoning yet.")
    print("Full training (50K+ steps) needed for quality outputs.")
    print("=" * 60)


if __name__ == "__main__":
    test_inference()
