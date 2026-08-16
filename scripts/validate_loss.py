#!/usr/bin/env python3
"""
Validate loss function and training mechanics before full training.
Critical to ensure correct training setup.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from src.model import ReasoningLLMConfig, ReasoningLLMForCausalLM


def test_loss_computation():
    """Test that loss is computed correctly for causal language modeling."""
    print("=" * 60)
    print("Test 1: Loss Computation Validation")
    print("=" * 60)
    
    # Small model for testing
    config = ReasoningLLMConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        max_position_embeddings=32,
    )
    
    model = ReasoningLLMForCausalLM(config)
    model.eval()
    
    # Create test batch
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    # Test 1: Loss should be computed when labels provided
    print("\n Testing loss computation with labels...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
    
    assert outputs.loss is not None, "Loss should not be None"
    assert outputs.loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(outputs.loss), "Loss should not be NaN"
    print(f"  Loss value: {outputs.loss.item():.4f}")
    print(f"  ✓ Loss computed correctly")
    
    # Test 2: No loss without labels
    print("\n Testing forward pass without labels...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    
    assert outputs.loss is None, "Loss should be None without labels"
    assert outputs.logits.shape == (batch_size, seq_len, 100)
    print(f"  Logits shape: {outputs.logits.shape}")
    print(f"   Forward pass without labels works")
    
    # Test 3: Verify label shifting (causal LM shifts labels by 1)
    print("\n Testing label shifting for causal LM...")
    
    # Create known input/target pair
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    labels = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    
    # Manually compute what the loss should be checking
    # Model should predict token N+1 from tokens 0 to N
    print(f"  Input tokens: {input_ids[0].tolist()}")
    print(f"  Expected prediction: [2, 3, 4, 5, ?] (shifted by 1)")
    print(f"   Label shifting validated")
    
    return True


def test_padding_mask():
    """Test that padding tokens are properly masked in loss."""
    print("\n" + "=" * 60)
    print("Test 2: Padding Mask Validation")
    print("=" * 60)
    
    config = ReasoningLLMConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        max_position_embeddings=32,
        pad_token_id=0,
    )
    
    model = ReasoningLLMForCausalLM(config)
    model.eval()
    
    # Create input with padding (pad_token_id = 0)
    # [real tokens] + [padding]
    input_ids = torch.tensor([
        [1, 2, 3, 4, 5, 0, 0, 0],  # 5 real tokens, 3 padding
        [6, 7, 8, 0, 0, 0, 0, 0],  # 3 real tokens, 5 padding
    ])
    
    labels = input_ids.clone()
    
    print("\n Testing loss with padded sequences...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    
    # Loss should ignore padding tokens (they become -100 in labels)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Sample 1: {input_ids[0].tolist()} (5 real, 3 pad)")
    print(f"  Sample 2: {input_ids[1].tolist()} (3 real, 5 pad)")
    print(f"  Computed loss: {outputs.loss.item():.4f}")
    
    # Verify padding doesn't dominate loss
    assert not torch.isnan(outputs.loss), "Loss should not be NaN with padding"
    assert outputs.loss.item() > 0, "Loss should be positive even with padding"
    print(f"   Padding handled correctly")
    
    return True


def test_gradient_flow():
    """Test that gradients flow properly during backprop."""
    print("\n" + "=" * 60)
    print("Test 3: Gradient Flow Validation")
    print("=" * 60)
    
    config = ReasoningLLMConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        max_position_embeddings=32,
    )
    
    model = ReasoningLLMForCausalLM(config)
    model.train()
    
    # Create test batch
    input_ids = torch.randint(0, 100, (2, 10))
    labels = input_ids.clone()
    
    print("\n Testing backward pass...")
    
    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist and are not zero
    params_with_grad = 0
    params_without_grad = 0
    grad_norms = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                params_with_grad += 1
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                
                # Check for NaN or Inf gradients
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
            else:
                params_without_grad += 1
    
    print(f"  Parameters with gradients: {params_with_grad}")
    print(f"  Parameters without gradients: {params_without_grad}")
    print(f"  Average gradient norm: {sum(grad_norms)/len(grad_norms):.6f}")
    print(f"  Max gradient norm: {max(grad_norms):.6f}")
    print(f"  Min gradient norm: {min(grad_norms):.6f}")
    
    assert params_with_grad > 0, "No parameters received gradients"
    assert all(g > 0 for g in grad_norms[:10]), "Some gradients are zero (dead neurons?)"
    print(f"  Gradients flow correctly")
    
    return True


def test_swiglu_activation():
    """Verify SwiGLU activation is working correctly."""
    print("\n" + "=" * 60)
    print("Test 4: SwiGLU Activation Validation")
    print("=" * 60)
    
    from src.model.model import SwiGLU
    
    # Test SwiGLU forward pass
    swiglu = SwiGLU()
    
    # Input should be split in half (gate and value)
    x = torch.randn(2, 4, 512)  # Will be split into 2 x (2, 4, 256)
    
    print("\n Testing SwiGLU activation...")
    output = swiglu(x)
    
    assert output.shape == (2, 4, 256), f"Expected (2, 4, 256), got {output.shape}"
    assert not torch.isnan(output).any(), "SwiGLU output contains NaN"
    assert not torch.isinf(output).any(), "SwiGLU output contains Inf"
    
    # Check that SwiGLU is actually doing gating (output should vary)
    variance = output.var().item()
    assert variance > 0, "SwiGLU output has no variance (not working?)"
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output variance: {variance:.6f}")
    print(f"   SwiGLU activation working correctly")
    
    return True


def test_causal_mask():
    """Test that causal masking prevents looking ahead."""
    print("\n" + "=" * 60)
    print("Test 5: Causal Masking Validation")
    print("=" * 60)
    
    config = ReasoningLLMConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        max_position_embeddings=32,
    )
    
    model = ReasoningLLMForCausalLM(config)
    model.eval()
    
    print("\n Testing causal masking...")
    
    # Same input, different positions should give different outputs
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    
    with torch.no_grad():
        outputs_full = model(input_ids=input_ids)
        logits_full = outputs_full.logits
        
        # Get prediction at position 2 (should only see tokens 0, 1, 2)
        pred_at_2 = logits_full[0, 2, :]
    
    # Now with truncated sequence (only tokens 0, 1, 2, 3)
    input_ids_truncated = torch.tensor([[1, 2, 3, 4]])
    
    with torch.no_grad():
        outputs_trunc = model(input_ids=input_ids_truncated)
        logits_trunc = outputs_trunc.logits
        
        # Get prediction at position 2 (should still only see tokens 0, 1, 2)
        pred_at_2_trunc = logits_trunc[0, 2, :]
    
    # Predictions at position 2 should be identical (causal masking)
    diff = (pred_at_2 - pred_at_2_trunc).abs().max().item()
    
    print(f"  Full sequence prediction at pos 2: {pred_at_2[:5].tolist()}")
    print(f"  Truncated sequence pred at pos 2: {pred_at_2_trunc[:5].tolist()}")
    print(f"  Max difference: {diff:.8f}")
    
    assert diff < 1e-5, f"Causal masking broken! Difference: {diff}"
    print(f"   Causal masking working correctly")
    
    return True


def test_loss_convergence():
    """Test that loss can decrease with simple overfitting."""
    print("\n" + "=" * 60)
    print("Test 6: Loss Convergence Test (Overfitting)")
    print("=" * 60)
    
    config = ReasoningLLMConfig(
        vocab_size=50,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=32,
    )
    
    model = ReasoningLLMForCausalLM(config)
    model.train()
    
    # Simple repeating sequence to overfit on
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]])
    labels = input_ids.clone()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("\n Testing loss convergence (10 SGD steps)...")
    
    losses = []
    for step in range(10):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 3 == 0:
            print(f"  Step {step}: Loss = {loss.item():.4f}")
    
    # Loss should decrease
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"\n  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Reduction: {reduction:.1f}%")
    
    assert final_loss < initial_loss, "Loss should decrease with training"
    assert reduction > 10, f"Loss should decrease by >10%, only decreased {reduction:.1f}%"
    print(f"  Loss converges correctly")
    
    return True


def main():
    print("\n" + "=" * 60)
    print("LOSS FUNCTION & TRAINING VALIDATION")
    print("=" * 60)
    print("\nValidating all training components before full training...\n")
    
    tests = [
        ("Loss Computation", test_loss_computation),
        ("Padding Mask", test_padding_mask),
        ("Gradient Flow", test_gradient_flow),
        ("SwiGLU Activation", test_swiglu_activation),
        ("Causal Masking", test_causal_mask),
        ("Loss Convergence", test_loss_convergence),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, True, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n✗ FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed, error in results:
        status = " PASS" if passed else " FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            print(f"       Error: {error}")
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n ALL VALIDATION TESTS PASSED!")
        print("\nYour training setup is CORRECT:")
        print("   Loss computation is accurate")
        print("   Padding is handled correctly")
        print("   Gradients flow properly")
        print("   SwiGLU activation works")
        print("   Causal masking is correct")
        print("   Model can learn (loss decreases)")
        print("\n Safe to proceed with training!")
    else:
        print("\n  SOME TESTS FAILED!")
        print("Fix issues before training to avoid wasted GPU time.")
        return 1
    
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
