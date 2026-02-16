# Quick Start Guide - Deep Reasoning LLM

This guide will help you get started with training your own deep reasoning language model.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended: 16GB+ VRAM for base model)
- 50GB+ free disk space

## Step 1: Setup Environment

### Create and activate virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 2: Prepare Your Data

You have two options:

### Option A: Use sample data (for testing)

```bash
python scripts/prepare_sample_data.py
```

### Option B: Use your own data

1. Create training data in JSONL format:
```json
{"text": "Your training text here"}
{"question": "What is...", "reasoning": "Let me think...", "answer": "The answer is..."}
```

2. Place files in:
   - `data/train/` - Training data
   - `data/eval/` - Evaluation data
   - `data/raw/` - Raw text for tokenizer

## Step 3: Train Tokenizer

```bash
python scripts/train_tokenizer.py \
    --data_dir data/raw \
    --output_dir tokenizer \
    --vocab_size 50000 \
    --model_type unigram
```

This will create:
- `tokenizer/tokenizer.model` - SentencePiece model
- `tokenizer/tokenizer.vocab` - Vocabulary file

## Step 4: Verify Model Configuration

Check the configuration file and estimated parameters:

```bash
python -c "from src.model import ReasoningLLMConfig; config = ReasoningLLMConfig(); config.print_config()"
```

Expected output:
```
============================================================
Deep Reasoning LLM Configuration
============================================================
Vocabulary Size:        50,000
Hidden Size:            768
Number of Layers:       12
Attention Heads:        12
Head Dimension:         64
FFN Intermediate Size:  3,072
Max Sequence Length:    512
Estimated Parameters:   110-125M
============================================================
```

## Step 5: Train the Model

### Basic training

```bash
python scripts/train.py \
    --config configs/base_config.yaml \
    --tokenizer_path tokenizer/tokenizer.model
```

### Training with custom settings

Edit `configs/base_config.yaml` to adjust:
- Batch size
- Learning rate
- Training steps
- Model dimensions

### Monitor training

```bash
# TensorBoard
tensorboard --logdir logs/

# Or use Weights & Biases (set wandb_enabled: true in config)
```

## Step 6: Generate Text (Inference)

### Single generation

```bash
python scripts/inference.py \
    --checkpoint checkpoints/checkpoint-10000 \
    --prompt "Let's solve 25 + 17 step by step:" \
    --max_length 200
```

### Interactive mode

```bash
python scripts/inference.py \
    --checkpoint checkpoints/checkpoint-10000 \
    --interactive
```

## Scaling Your Model

To train larger models, use the scaling presets:

```bash
# Copy scaling preset to your config
cp configs/scaling_presets.yaml configs/my_config.yaml

# Edit configs/my_config.yaml to select model size
# Then train with that config
python scripts/train.py --config configs/my_config.yaml
```

Available sizes:
- **Small**: ~110M params (base config)
- **Medium**: ~350M params
- **Large**: ~760M params
- **XLarge**: ~1.3B params
- **XXLarge**: ~2.7B params

## Tips for Best Results

### Data Quality
- Use high-quality, diverse datasets
- Include reasoning examples (chain-of-thought)
- Mix different data types (text, code, math)
- Clean and deduplicate your data

### Training
- Start with a smaller model to validate your pipeline
- Use mixed precision (BF16) to save memory
- Monitor validation loss to prevent overfitting
- Save checkpoints frequently

### Hardware Requirements by Model Size

| Model Size | Min VRAM | Recommended | Batch Size |
|------------|----------|-------------|------------|
| Small (110M) | 8GB | 16GB | 32 |
| Medium (350M) | 16GB | 24GB | 16 |
| Large (760M) | 24GB | 40GB | 8 |
| XLarge (1.3B) | 40GB | 80GB | 4 |

### Distributed Training

For larger models, use distributed training:

```bash
# Single machine, multiple GPUs
torchrun --nproc_per_node=4 scripts/train.py --config configs/base_config.yaml
```

## Recommended Datasets

For production-quality training:

1. **General Text**:
   - OpenWebText
   - The Pile
   - C4 (Colossal Clean Crawled Corpus)

2. **Reasoning**:
   - GSM8K (math word problems)
   - MATH dataset
   - StrategyQA
   - CommonsenseQA

3. **Code**:
   - The Stack
   - CodeParrot
   - GitHub Code

4. **Chain-of-Thought**:
   - CoT datasets from Google Research
   - Self-instruct datasets
   - OpenOrca

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size
- Increase gradient accumulation steps
- Use smaller sequence length
- Enable gradient checkpointing

### Slow Training
- Increase number of data workers
- Use SSD for data storage
- Enable mixed precision training
- Consider using Flash Attention

### Poor Performance
- Train longer (more steps)
- Increase model size
- Improve data quality
- Adjust learning rate

## Next Steps

1. **Experiment with prompts**: Test different reasoning prompts
2. **Fine-tune for specific tasks**: Adapt the model to your use case
3. **Evaluate**: Create benchmarks to measure reasoning ability
4. **Iterate**: Continuously improve data and training setup

## Support and Resources

- Configuration reference: See `configs/base_config.yaml`
- Scaling guide: See `configs/scaling_presets.yaml`
- Model architecture: See `src/model/model.py`
- Training code: See `src/training/trainer.py`

Happy training! 🚀
