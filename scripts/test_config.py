#!/usr/bin/env python3
"""
Test script to verify model configuration and setup.
Run this after installation to ensure everything is working.

Usage:
    python scripts/test_config.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.model import ReasoningLLMConfig, ReasoningLLMForCausalLM
        from src.utils import set_seed, count_parameters
        print("yep Model imports successful")
    except ImportError as e:
        print(f"nope Import error: {e}")
        return False
    
    try:
        import transformers
        import sentencepiece
        print("✓ External dependencies available")
    except ImportError as e:
        print(f"nope Missing dependency: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model creation and parameter counting."""
    print("\nTesting model creation...")
    
    try:
        from src.model import ReasoningLLMConfig, ReasoningLLMForCausalLM
        from src.utils import count_parameters
        
        # Create config
        config = ReasoningLLMConfig(
            vocab_size=50000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
        )
        
        print("\nModel Configuration:")
        config.print_config()
        
        # Create model
        model = ReasoningLLMForCausalLM(config)
        
        # Count parameters
        params = count_parameters(model)
        
        print(f"\nActual Parameters:")
        print(f"  Total: {params['total']:,} ({params['total']/1e6:.2f}M)")
        print(f"  Trainable: {params['trainable']:,}")
        
        # Verify parameter count is in expected range
        expected_min = 110_000_000
        expected_max = 125_000_000
        
        if expected_min <= params['total'] <= expected_max:
            print(f"yep Parameter count is in expected range ({expected_min/1e6:.0f}M - {expected_max/1e6:.0f}M)")
        else:
            print(f"⚠ Warning: Parameter count outside expected range")
        
        return True
        
    except Exception as e:
        print(f"nope Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test a forward pass through the model."""
    print("\nTesting forward pass...")
    
    try:
        from src.model import ReasoningLLMConfig, ReasoningLLMForCausalLM
        
        # Create small model for testing
        config = ReasoningLLMConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
            max_position_embeddings=64,
        )
        
        model = ReasoningLLMForCausalLM(config)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        # Check output shape
        expected_shape = (batch_size, seq_length, 1000)
        if outputs.logits.shape == expected_shape:
            print(f"yep Forward pass successful")
            print(f"  Input shape: {input_ids.shape}")
            print(f"  Output shape: {outputs.logits.shape}")
            return True
        else:
            print(f"nope Unexpected output shape: {outputs.logits.shape}")
            return False
            
    except Exception as e:
        print(f"nope Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_device():
    """Test CUDA availability."""
    print("\nTesting device availability...")
    
    if torch.cuda.is_available():
        print(f"yep CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("nope CUDA not available - will use CPU")
        print("  Training on CPU will be very slow")
        print("  Consider using a GPU for training")
    
    if torch.backends.mps.is_available():
        print("yep MPS (Apple Silicon) available")
    
    return True


def test_configuration_files():
    """Test that configuration files exist."""
    print("\nTesting configuration files...")
    
    config_files = [
        "configs/base_config.yaml",
        "configs/scaling_presets.yaml",
        "requirements.txt",
        "README.md",
        "QUICKSTART.md",
    ]
    
    all_exist = True
    for config_file in config_files:
        path = Path(config_file)
        if path.exists():
            print(f"yes {config_file}")
        else:
            print(f"nope Missing: {config_file}")
            all_exist = False
    
    return all_exist


def main():
    print("=" * 60)
    print("Deep Reasoning LLM - Configuration Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Files", test_configuration_files),
        ("Device Test", test_device),
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"{test_name}")
        print("=" * 60)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Prepare your training data or use sample data:")
        print("   python scripts/prepare_sample_data.py")
        print("\n2. Train a tokenizer:")
        print("   python scripts/train_tokenizer.py --data_dir data/raw --output_dir tokenizer")
        print("\n3. Start training:")
        print("   python scripts/train.py --config configs/base_config.yaml")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        print("Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
