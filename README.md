# Deep Reasoning LLM

A state-of-the-art transformer-based language model focused on deep reasoning and chain-of-thought capabilities.

## Model Specifications

- **Parameters**: ~190M (with SwiGLU activation)
- **Architecture**: Transformer with scalable configuration
- **Hidden Dimension**: 768
- **Layers**: 12
- **Attention Heads**: 12  
- **FFN Size**: 3072
- **Vocabulary Size**: 50,000
- **Context Length**: 512
- **Tokenizer**: SentencePiece (Unigram)
- **Activation**: SwiGLU (state-of-the-art)

```
llm_self_attention/
├── configs/              # Model and training configurations
├── src/                  # Source code
│   ├── model/           # Model architecture
│   ├── data/            # Data processing and loading
│   ├── tokenizer/       # Tokenizer setup
│   ├── training/        # Training logic
│   └── utils/           # Utility functions
├── scripts/             # Training and evaluation scripts
├── data/                # Dataset storage
├── checkpoints/         # Model checkpoints
├── logs/                # Training logs
└── tests/               # Unit tests
```

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train tokenizer:
```bash
python scripts/train_tokenizer.py --data_path data/raw --output_path tokenizer/
```

4. Train model:
```bash
python scripts/train.py --config configs/base_config.yaml
```

## Features

- ✅ Scalable transformer architecture
- ✅ SentencePiece tokenizer with Unigram algorithm
- ✅ SwiGLU activation (used in LLaMA, PaLM)
- ✅ RoPE positional embeddings
- ✅ Flexible configuration system
- ✅ Mixed precision training (FP16/BF16)
- ✅ Gradient accumulation
- ✅ Distributed training support
- ✅ Chain-of-thought training capabilities
- ✅ Comprehensive logging and monitoring

## License

MIT License
