"""
Utility functions for Deep Reasoning LLM.
"""

import random
import numpy as np
import torch
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Union


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif path.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False)
        elif path.suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def model_summary(model: torch.nn.Module):
    """
    Print model summary with layer details.
    
    Args:
        model: PyTorch model
    """
    print("=" * 80)
    print("Model Summary")
    print("=" * 80)
    
    total_params = 0
    trainable_params = 0
    
    print(f"{'Layer':<40} {'Output Shape':<20} {'Params':>15}")
    print("-" * 80)
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            total_params += params
            trainable_params += trainable
            
            if params > 0:
                print(f"{name:<40} {'':<20} {params:>15,}")
    
    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    print("=" * 80)


def calculate_flops_per_token(
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    vocab_size: int,
    seq_length: int,
) -> int:
    """
    Estimate FLOPs per token for transformer model.
    
    Args:
        num_layers: Number of transformer layers
        hidden_size: Hidden dimension size
        num_heads: Number of attention heads
        vocab_size: Vocabulary size
        seq_length: Sequence length
        
    Returns:
        Estimated FLOPs per token
    """
    # Attention: Q, K, V projections + output projection
    attn_flops = 4 * hidden_size * hidden_size
    
    # Attention computation
    attn_compute = 2 * seq_length * hidden_size
    
    # FFN: 2 linear layers
    ffn_flops = 8 * hidden_size * hidden_size
    
    # Per layer
    per_layer_flops = attn_flops + attn_compute + ffn_flops
    
    # All layers
    total_flops = num_layers * per_layer_flops
    
    # Add embedding and output layer
    total_flops += 2 * hidden_size * vocab_size
    
    return total_flops


if __name__ == "__main__":
    # Test utilities
    set_seed(42)
    print("✓ Seed set")
    
    # Test parameter counting
    test_model = torch.nn.Linear(100, 10)
    params = count_parameters(test_model)
    print(f"✓ Test model parameters: {params}")
    
    # Test time formatting
    print(f"✓ Format time: {format_time(3661)}")
