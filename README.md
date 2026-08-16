<<<<<<< HEAD
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

-  Scalable transformer architecture
-  SentencePiece tokenizer with Unigram algorithm
-  SwiGLU activation (used in LLaMA, PaLM)
-  RoPE positional embeddings
-  Flexible configuration system
-  Mixed precision training (FP16/BF16)
-  Gradient accumulation
-  Distributed training support
-  Chain-of-thought training capabilities
-  Comprehensive logging and monitoring

## License

MIT License
=======
# llm_self-attention
## created a 128million parameters thinking model from scratch with training from scratch loss convergence moving average plates to 1.9 on 25k 8 batch training. this is a scalable upto billions of parameters just change the configuration in base-config.yaml  &amp; use scaling_preset.yaml remember the model needs more training since we use batched randomization so that modal generalizes and doesnt over fits despite having small number of parameters. we keep the number of paramaters small intentionally since the modal needs validation on every stage so that we can predict the modal and disciver its uniwue scaling laws. since modifying modal archeture when its smaller is easier than modifying its arch when the modal has grown larger. i suggest to push the parameters to 300-500 million ranger if your machine can handle it. 
>>>>>>> 7cc2b760938a382bd277aaca66f6b679900efe50
