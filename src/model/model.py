"""
Core model architecture for Deep Reasoning LLM.
Built on transformer architecture with optimizations for reasoning tasks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin

from .config import ReasoningLLMConfig


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for better position encoding.
    More scalable than learned positional embeddings.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate position indices
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to queries and keys."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) with Flash Attention 2 support.
    
    GQA reduces memory and improves inference speed by sharing K/V heads across Q heads.
    Uses PyTorch's scaled_dot_product_attention (SDPA) for Flash Attention 2.
    
    Architecture used in: Llama 3, Mistral, Gemma, Phi-3 (2024-2026 SOTA models)
    
    Args:
        config: Model configuration with num_attention_heads and num_key_value_heads
    """
    def __init__(self, config: ReasoningLLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.dropout = config.attention_probs_dropout_prob
        self.use_flash_attention = config.use_flash_attention
        
        # Query projection (full number of heads)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        
        # Key/Value projections (fewer heads for GQA)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Rotary embeddings (RoPE)
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim, 
            max_position_embeddings=config.max_position_embeddings
        )
        
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat K/V heads to match the number of Q heads for GQA.
        
        This is the core of GQA: each KV head is shared across multiple Q heads.
        For example, with 12 Q heads and 4 KV heads, each KV head is repeated 3 times.
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE (Rotary Position Embeddings)
        cos, sin = self.rotary_emb(value_states, seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Handle KV cache for efficient generation
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # GQA: Repeat K/V heads to match Q heads
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # Flash Attention 2 (via PyTorch SDPA - works on CPU/CUDA/MPS)
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Handle attention mask for SDPA
            # SDPA expects: (batch, num_heads, query_len, key_len)
            # During generation with KV cache: query_len=1, key_len grows
            
            batch_size, num_heads, query_len, _ = query_states.shape
            _, _, key_len, _ = key_states.shape
            
            if attention_mask is not None:
                # attention_mask shape: (batch, 1, query_len, key_len)
                # Ensure it matches the actual key length
                mask_key_len = attention_mask.shape[-1]
                
                if mask_key_len != key_len:
                    # During generation, mask might be for shorter sequence
                    # Create proper causal mask for current state
                    attn_mask = torch.zeros(
                        (batch_size, 1, query_len, key_len),
                        dtype=query_states.dtype,
                        device=query_states.device
                    )
                    # Allow attending to all past tokens (KV cache) and current token
                    # No masking needed - all positions are valid
                else:
                    attn_mask = attention_mask
            else:
                attn_mask = None
            
            # Use PyTorch's optimized SDPA (includes Flash Attention on CUDA)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,  # We handle causality via attention_mask
            )
        else:
            # Fallback: Manual attention computation (original implementation)
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, past_key_value


class SwiGLU(nn.Module):
    """
    SwiGLU activation function: SwiGLU(x) = Swish(Wx) ⊙ Vx
    More effective than GELU, used in LLaMA, PaLM, and other SOTA models.
    Paper: https://arxiv.org/abs/2002.05202
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with SwiGLU activation.
    Uses gated linear units for better performance than standard FFN.
    """
    def __init__(self, config: ReasoningLLMConfig):
        super().__init__()
        # SwiGLU requires 2x intermediate size for gate and up projections
        # We split the output in SwiGLU, so we use 2 * intermediate_size
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.swiglu = SwiGLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Combined gate and up projection
        x = self.gate_up_proj(x)
        # Apply SwiGLU activation
        x = self.swiglu(x)
        x = self.dropout(x)
        # Down projection
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-LayerNorm architecture.
    Uses residual connections and layer normalization.
    """
    def __init__(self, config: ReasoningLLMConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = FeedForward(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, past_key_value = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output
        
        # Feed-forward with residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        ff_output = self.mlp(hidden_states)
        hidden_states = residual + ff_output
        
        return hidden_states, past_key_value


class ReasoningLLMPreTrainedModel(PreTrainedModel):
    """
    Base class for all models in this project.
    Handles weight initialization.
    """
    config_class = ReasoningLLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class ReasoningLLMModel(ReasoningLLMPreTrainedModel):
    """
    The main transformer model without the language modeling head.
    This is the base model that can be extended for different tasks.
    """
    def __init__(self, config: ReasoningLLMConfig):
        super().__init__(config)
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        # Transformer layers
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize weights
        self.post_init()
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ):
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        batch_size, seq_length = input_ids.shape
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=input_ids.device)
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones((seq_length, seq_length), dtype=torch.bool, device=input_ids.device), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
        # Convert to boolean for bitwise operations
        attention_mask = attention_mask.bool() & ~causal_mask
        # Convert to float for attention computation
        attention_mask = torch.where(attention_mask, 0.0, float('-inf'))
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)
        
        # Pass through transformer layers
        all_hidden_states = () if output_hidden_states else None
        next_cache = () if use_cache else None
        
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Handle both tuple and DynamicCache types
            if past_key_values is not None:
                if hasattr(past_key_values, '__getitem__'):
                    past_key_value = past_key_values[idx]
                else:
                    # DynamicCache or similar object
                    past_key_value = None
            else:
                past_key_value = None
            
            hidden_states, present_key_value = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            if use_cache:
                next_cache = next_cache + (present_key_value,)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states] if v is not None)
        
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': next_cache,
            'hidden_states': all_hidden_states,
        }


class ReasoningLLMForCausalLM(ReasoningLLMPreTrainedModel, GenerationMixin):
    """
    ReasoningLLM Model with a language modeling head for causal language modeling.
    This is the main model for training and generation.
    Uses SwiGLU activation for improved performance.
    """
    def __init__(self, config: ReasoningLLMConfig):
        super().__init__(config)
        self.model = ReasoningLLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights between input embeddings and output layer (optional)
        # self.lm_head.weight = self.model.embed_tokens.weight
        
        # Initialize weights
        self.post_init()
        
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward pass through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        hidden_states = outputs['last_hidden_state']
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,) + tuple(v for k, v in outputs.items() if k != 'last_hidden_state')
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.get('past_key_values'),
            hidden_states=outputs.get('hidden_states'),
        )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        """Prepare inputs for generation (used during inference)."""
        if past_key_values:
            input_ids = input_ids[:, -1:]
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
        
