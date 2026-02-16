# Makefile for Deep Reasoning LLM

.PHONY: help install test clean data tokenizer train inference

help:
	@echo "Deep Reasoning LLM - Available Commands"
	@echo "========================================"
	@echo "make install      - Install dependencies"
	@echo "make test         - Test configuration"
	@echo "make data         - Generate sample data"
	@echo "make tokenizer    - Train tokenizer"
	@echo "make train        - Start training"
	@echo "make inference    - Run inference (interactive)"
	@echo "make clean        - Clean generated files"
	@echo "========================================"

install:
	@echo "Installing dependencies..."
	python3 -m pip install --upgrade pip
	python3 -m pip install -r requirements.txt
	@echo "✓ Installation complete"

test:
	@echo "Running configuration tests..."
	python3 scripts/test_config.py

data:
	@echo "Generating sample data..."
	python3 scripts/prepare_sample_data.py

tokenizer:
	@echo "Training tokenizer..."
	python3 scripts/train_tokenizer.py \
		--data_dir data/raw \
		--output_dir tokenizer \
		--vocab_size 50000 \
		--model_type unigram

train:
	@echo "Starting training..."
	python3 scripts/train.py \
		--config configs/base_config.yaml \
		--tokenizer_path tokenizer/tokenizer.model

inference:
	@echo "Starting interactive inference..."
	python3 scripts/inference.py \
		--checkpoint checkpoints/checkpoint-10000 \
		--tokenizer_path tokenizer/tokenizer.model \
		--interactive

clean:
	@echo "Cleaning generated files..."
	rm -rf checkpoints/
	rm -rf logs/
	rm -rf wandb/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleanup complete"

# Quick workflow
quickstart: install data tokenizer test
	@echo ""
	@echo "========================================"
	@echo "Quick start setup complete!"
	@echo "========================================"
	@echo "Run 'make train' to start training"
	@echo "========================================"
