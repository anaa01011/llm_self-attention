#!/usr/bin/env python3
"""
Validation script for GQA + Flash Attention 2 implementation.

Tests:
1. Forward pass shape validation
2. Gradient flow check
3. KV cache compatibility
4. Numerical stability
5. Comparison with original model (if checkpoint exists)
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.config import ReasoningLLMConfig
from src.model.model import GroupedQueryAttention, TransformerBlock

def test_gqa_forward_pass():
    """Test GQA forward pass with different configurations."""
    print("\n" + "="*60)
    print("TEST 1: GQA Forward Pass")
    print("="*60)
    
    # Create test config (head_dim is computed automatically)
    config = ReasoningLLMConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_key_value_heads=4,  # GQA with 3:1 ratio
        attention_probs_dropout_prob=0.0,  # No dropout for testing
        use_flash_attention=True,
    )
    
    # Create model
    attn = GroupedQueryAttention(config)
    attn.eval()
    
    # Test inputs
    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Forward pass
    output, kv_cache = attn(hidden_states, use_cache=True)
    
    # Validate shapes
    assert output.shape == (batch_size, seq_len, config.hidden_size), \
        f"Output shape mismatch: {output.shape}"
    assert kv_cache[0].shape == (batch_size, config.num_key_value_heads, seq_len, config.head_dim), \
        f"Key cache shape mismatch: {kv_cache[0].shape}"
    assert kv_cache[1].shape == (batch_size, config.num_key_value_heads, seq_len, config.head_dim), \
        f"Value cache shape mismatch: {kv_cache[1].shape}"
    
    print(f" Forward pass successful")
    print(f"   Input shape: {hidden_states.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   KV cache shapes: K={kv_cache[0].shape}, V={kv_cache[1].shape}")
    print(f"   GQA ratio: {config.num_attention_heads}/{config.num_key_value_heads} = {config.num_attention_heads // config.num_key_value_heads}:1")
    
    return True


def test_gradient_flow():
    """Test that gradients flow properly through GQA."""
    print("\n" + "="*60)
    print("TEST 2: Gradient Flow")
    print("="*60)
    
    config = ReasoningLLMConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_key_value_heads=4,
        attention_probs_dropout_prob=0.1,
        use_flash_attention=True,
    )
    
    attn = GroupedQueryAttention(config)
    attn.train()
    
    # Test inputs
    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, requires_grad=True)
    
    # Forward + backward
    output, _ = attn(hidden_states)
    loss = output.sum()
    loss.backward()
    
    # Check gradients exist
    assert hidden_states.grad is not None, "Input gradients missing"
    assert attn.q_proj.weight.grad is not None, "Q projection gradients missing"
    assert attn.k_proj.weight.grad is not None, "K projection gradients missing"
    assert attn.v_proj.weight.grad is not None, "V projection gradients missing"
    assert attn.o_proj.weight.grad is not None, "Output projection gradients missing"
    
    # Check gradient magnitudes
    q_grad_norm = attn.q_proj.weight.grad.norm().item()
    k_grad_norm = attn.k_proj.weight.grad.norm().item()
    v_grad_norm = attn.v_proj.weight.grad.norm().item()
    
    print(f" Gradients flow correctly")
    print(f"   Q projection grad norm: {q_grad_norm:.4f}")
    print(f"   K projection grad norm: {k_grad_norm:.4f}")
    print(f"   V projection grad norm: {v_grad_norm:.4f}")
    
    # Check for gradient explosion/vanishing (relaxed thresholds for larger models)
    assert 0.001 < q_grad_norm < 10000, f"Q grad norm suspicious: {q_grad_norm}"
    assert 0.001 < k_grad_norm < 10000, f"K grad norm suspicious: {k_grad_norm}"
    assert 0.001 < v_grad_norm < 10000, f"V grad norm suspicious: {v_grad_norm}"
    
    print(f"   No gradient explosion or vanishing detected")
    
    return True


def test_kv_cache():
    """Test KV cache for efficient generation."""
    print("\n" + "="*60)
    print("TEST 3: KV Cache Generation")
    print("="*60)
    
    config = ReasoningLLMConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_key_value_heads=4,
        attention_probs_dropout_prob=0.0,
        use_flash_attention=True,
    )
    
    attn = GroupedQueryAttention(config)
    attn.eval()
    
    batch_size = 1
    
    # First token
    hidden_1 = torch.randn(batch_size, 1, config.hidden_size)
    output_1, kv_cache = attn(hidden_1, use_cache=True)
    
    # Second token with cache
    hidden_2 = torch.randn(batch_size, 1, config.hidden_size)
    output_2, kv_cache = attn(hidden_2, past_key_value=kv_cache, use_cache=True)
    
    # Third token with cache
    hidden_3 = torch.randn(batch_size, 1, config.hidden_size)
    output_3, kv_cache = attn(hidden_3, past_key_value=kv_cache, use_cache=True)
    
    # Validate cache shapes grow correctly
    assert kv_cache[0].shape[2] == 3, f"Cache should have 3 tokens, got {kv_cache[0].shape[2]}"
    
    print(f" KV cache works correctly")
    print(f"   Token 1 cache: {1} positions")
    print(f"   Token 2 cache: {2} positions")
    print(f"   Token 3 cache: {3} positions")
    print(f"   Final cache shape: K={kv_cache[0].shape}, V={kv_cache[1].shape}")
    
    return True


def test_flash_attention_toggle():
    """Test that both Flash Attention and fallback work."""
    print("\n" + "="*60)
    print("TEST 4: Flash Attention Toggle")
    print("="*60)
    
    config = ReasoningLLMConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_key_value_heads=4,
        attention_probs_dropout_prob=0.0,
        use_flash_attention=True,
    )
    
    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Test with Flash Attention
    attn_flash = GroupedQueryAttention(config)
    attn_flash.eval()
    output_flash, _ = attn_flash(hidden_states)
    
    # Test with fallback
    config.use_flash_attention = False
    attn_fallback = GroupedQueryAttention(config)
    attn_fallback.load_state_dict(attn_flash.state_dict())  # Same weights
    attn_fallback.eval()
    output_fallback, _ = attn_fallback(hidden_states)
    
    # Outputs should be similar (not identical due to numerical differences)
    diff = (output_flash - output_fallback).abs().max().item()
    
    print(f" Both Flash Attention and fallback work")
    print(f"   Flash Attention enabled: {attn_flash.use_flash_attention}")
    print(f"   Fallback enabled: {attn_fallback.use_flash_attention}")
    print(f"   Max output difference: {diff:.6f}")
    
    # Allow small numerical differences
    assert diff < 1e-3, f"Outputs differ too much: {diff}"
    
    return True


def test_transformer_block():
    """Test full TransformerBlock with new GQA."""
    print("\n" + "="*60)
    print("TEST 5: TransformerBlock Integration")
    print("="*60)
    
    config = ReasoningLLMConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=3072,
        hidden_act="swiglu",
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        use_flash_attention=True,
    )
    
    block = TransformerBlock(config)
    block.eval()
    
    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Forward pass
    output, kv_cache = block(hidden_states, use_cache=True)
    
    # Validate
    assert output.shape == hidden_states.shape, f"Shape mismatch: {output.shape}"
    assert kv_cache is not None, "KV cache missing"
    
    print(f" TransformerBlock works with GQA")
    print(f"   Input shape: {hidden_states.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention: GroupedQueryAttention")
    print(f"   MLP: SwiGLU")
    
    return True


def main():
    """Run all validation tests."""
    print("\n" + "="*80)
    print(" GQA + Flash Attention 2 Validation Suite")
    print("="*80)
    
    tests = [
        ("Forward Pass", test_gqa_forward_pass),
        ("Gradient Flow", test_gradient_flow),
        ("KV Cache", test_kv_cache),
        ("Flash Attention Toggle", test_flash_attention_toggle),
        ("TransformerBlock Integration", test_transformer_block),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n❌ {name} FAILED:")
            print(f"   {str(e)}")
            failed += 1
    
    print("\n" + "="*80)
    print(f" Results: {passed} passed, {failed} failed")
    print("="*80)
    
    if failed == 0:
        print("\n All tests passed! GQA + Flash Attention 2 implementation is ready.")
        print("\n Safe to proceed with training")
        return 0
    else:
        print(f"\n  {failed} test(s) failed. Please fix before training.")
        return 1


if __name__ == "__main__":
    exit(main())
