"""
Configuration class for the Deep Reasoning LLM model.
Designed to be highly scalable and compatible with HuggingFace ecosystem.
"""

from typing import Optional
from transformers import PretrainedConfig


class ReasoningLLMConfig(PretrainedConfig):
    """
    Configuration class for ReasoningLLM model.
    
    This configuration is designed to be easily scalable - you can adjust parameters
    without changing the architecture code.
    """
    
    model_type = "reasoning_llm"
    
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: Optional[int] = None,  # GQA: if None, uses num_attention_heads (MHA)
        intermediate_size: int = 3072,
        hidden_act: str = "swiglu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        use_cache: bool = True,
        # Reasoning-specific parameters
        enable_reasoning_tokens: bool = True,
        reasoning_depth: int = 3,
        # Scaling parameters
        rope_scaling: Optional[dict] = None,
        use_flash_attention: bool = True,  # Use SDPA (Flash Attention) by default
        **kwargs
    ):
        """
        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden dimension size
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of query attention heads per layer
            num_key_value_heads: Number of KV heads for GQA (if None, uses num_attention_heads for MHA)
            intermediate_size: FFN intermediate dimension
            hidden_act: Activation function (swiglu recommended)
            hidden_dropout_prob: Dropout probability for hidden layers
            attention_probs_dropout_prob: Dropout probability for attention
            max_position_embeddings: Maximum sequence length
            initializer_range: Standard deviation for weight initialization
            layer_norm_eps: Epsilon for layer normalization
            pad_token_id: Padding token ID
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            use_cache: Whether to use KV cache
            enable_reasoning_tokens: Enable special reasoning tokens
            reasoning_depth: Maximum depth for chain-of-thought
            rope_scaling: RoPE scaling configuration (for future scaling)
            use_flash_attention: Use SDPA Flash Attention (PyTorch 2.0+)
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        # GQA: If num_key_value_heads is None, use MHA (all heads)
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        
        # Reasoning-specific
        self.enable_reasoning_tokens = enable_reasoning_tokens
        self.reasoning_depth = reasoning_depth
        
        # Scaling features (for future expansion)
        self.rope_scaling = rope_scaling
        self.use_flash_attention = use_flash_attention
        
        # Validate configuration
        assert hidden_size % num_attention_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
        assert num_attention_heads % self.num_key_value_heads == 0, \
            f"num_attention_heads ({num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
    
    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads
    
    def estimate_parameters(self) -> int:
        """
        Estimate total number of parameters in the model.
        Useful for verification before training.
        """
        # Embedding layer
        embedding_params = self.vocab_size * self.hidden_size
        
        # Position embeddings
        position_params = self.max_position_embeddings * self.hidden_size
        
        # Transformer layers
        per_layer_params = (
            # Self-attention: Q, K, V projections + output projection
            4 * (self.hidden_size * self.hidden_size) +
            # Layer norms (2 per layer)
            2 * (2 * self.hidden_size) +
            # FFN with SwiGLU: gate_up projection (2x intermediate) + down projection
            (self.hidden_size * 2 * self.intermediate_size) +
            (self.intermediate_size * self.hidden_size)
        )
        transformer_params = self.num_hidden_layers * per_layer_params
        
        # Final layer norm + output layer
        final_params = (2 * self.hidden_size) + (self.hidden_size * self.vocab_size)
        
        total = embedding_params + position_params + transformer_params + final_params
        
        return total
    
    def print_config(self):
        """Print configuration summary."""
        print("=" * 60)
        print("Deep Reasoning LLM Configuration")
        print("=" * 60)
        print(f"Vocabulary Size:        {self.vocab_size:,}")
        print(f"Hidden Size:            {self.hidden_size:,}")
        print(f"Number of Layers:       {self.num_hidden_layers}")
        print(f"Attention Heads:        {self.num_attention_heads}")
        print(f"Head Dimension:         {self.head_dim}")
        print(f"FFN Intermediate Size:  {self.intermediate_size:,}")
        print(f"Max Sequence Length:    {self.max_position_embeddings}")
        print(f"Estimated Parameters:   {self.estimate_parameters() / 1e6:.2f}M")
        print("=" * 60)


if __name__ == "__main__":
    # Test configuration
    config = ReasoningLLMConfig()
    config.print_config()
