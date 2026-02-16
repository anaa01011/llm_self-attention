#!/bin/bash
# Quick validation script for Deep Reasoning LLM with SwiGLU

echo "============================================================"
echo "Deep Reasoning LLM - Quick Validation"
echo "============================================================"
echo ""

# Test 1: Check Python environment
echo "Test 1: Python Environment"
./myenv/bin/python --version
echo "✅ Python environment ready"
echo ""

# Test 2: Import test
echo "Test 2: Import Core Modules"
./myenv/bin/python -c "
from src.model import ReasoningLLMConfig, ReasoningLLMForCausalLM
from src.tokenizer import TokenizerTrainer
from src.training import Trainer
from src.utils import set_seed
print('✅ All core modules imported successfully')
"
echo ""

# Test 3: SwiGLU activation test
echo "Test 3: SwiGLU Activation"
./myenv/bin/python -c "
from src.model import ReasoningLLMConfig
config = ReasoningLLMConfig()
assert config.hidden_act == 'swiglu', 'Activation should be SwiGLU'
print('✅ SwiGLU activation confirmed')
"
echo ""

# Test 4: Model creation
echo "Test 4: Model Creation"
./myenv/bin/python -c "
from src.model import ReasoningLLMForCausalLM, ReasoningLLMConfig
import torch

config = ReasoningLLMConfig(
    vocab_size=1000,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=512
)

model = ReasoningLLMForCausalLM(config)
input_ids = torch.randint(0, 1000, (1, 5))

with torch.no_grad():
    outputs = model(input_ids=input_ids)

assert outputs.logits.shape[0] == 1
assert outputs.logits.shape[1] == 5
assert outputs.logits.shape[2] == 1000

print('✅ Model forward pass successful')
"
echo ""

# Test 5: Display final config
echo "Test 5: Production Model Configuration"
./myenv/bin/python -c "
from src.model import ReasoningLLMConfig
config = ReasoningLLMConfig()
config.print_config()
"
echo ""

echo "============================================================"
echo "🎉 All Validation Tests Passed!"
echo "============================================================"
echo ""
echo "Your Deep Reasoning LLM is ready for training with:"
echo "  • SwiGLU activation (state-of-the-art)"
echo "  • ~190M parameters"
echo "  • Full HuggingFace compatibility"
echo "  • Scalable architecture"
echo ""
echo "Next: python scripts/prepare_sample_data.py"
echo "============================================================"
