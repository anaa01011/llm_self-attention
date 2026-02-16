#!/usr/bin/env python3
"""
Expert Iteration Training Script

Trains the model using expert iteration:
1. Generate K samples per question
2. Verify and score each sample
3. Train on top-quality samples only 
4. Auto-switch from dataset to automatic mode

10x faster learning than supervised training.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Dataset
from transformers import get_scheduler
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import ReasoningLLMForCausalLM, ReasoningLLMConfig
from src.tokenizer.tokenizer import ReasoningLLMTokenizer
from src.training.expert_iteration import ExpertIterationTrainer, ExpertIterationConfig
from src.training.verifier import ReasoningVerifier, MathVerifier


class RawTextDataset(Dataset):
    """Dataset that groups text into 512-token chunks for efficient training"""
    
    def __init__(self, data_path: str, tokenizer, max_examples: int = None, skip_examples: int = 0, max_tokens: int = 512):
        self.examples = []
        self.max_tokens = max_tokens
        
        # Load JSONL data
        data_file = Path(data_path)
        if data_file.is_file():
            files = [data_file]
        else:
            files = list(data_file.glob("**/*.jsonl"))
        
        # Load all individual lines first
        all_lines = []
        skipped = 0
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        # Skip first N examples
                        if skipped < skip_examples:
                            skipped += 1
                            continue
                            
                        example = json.loads(line)
                        if 'text' in example:
                            all_lines.append(example['text'])
                            # Stop if max_examples reached
                            if max_examples and len(all_lines) >= max_examples:
                                break
            if max_examples and len(all_lines) >= max_examples:
                break
        
        # Group lines into chunks that fit within max_tokens
        current_chunk = []
        current_tokens = 0
        
        for line_text in all_lines:
            # Tokenize to check length
            tokens = tokenizer.encode(line_text)
            line_token_count = len(tokens)
            
            # If adding this line exceeds max_tokens, save current chunk and start new one
            if current_tokens + line_token_count > max_tokens and current_chunk:
                combined_text = "\n\n".join(current_chunk)
                self.examples.append(combined_text)
                current_chunk = []
                current_tokens = 0
            
            # Add line to current chunk
            current_chunk.append(line_text)
            current_tokens += line_token_count
        
        # Add remaining chunk if any
        if current_chunk:
            combined_text = "\n\n".join(current_chunk)
            self.examples.append(combined_text)
        
        print(f"Loaded {len(all_lines)} lines, grouped into {len(self.examples)} chunks of ~{max_tokens} tokens (skipped first {skip_examples})")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {'text': self.examples[idx]}


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('expert_iteration_training.log')
    ]
)
logger = logging.getLogger(__name__)


def extract_question_and_answer(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract question, thinking, and conclusion from dataset text.
    
    Args:
        text: Full text from dataset
        
    Returns:
        (question, thinking, conclusion) tuple
    """
    # Pattern: "Question: ... <think>...</think> optional conclusion tags"
    question_match = re.search(r'Question:(.*?)(?:<think>|$)', text, re.DOTALL)
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    conclusion_match = re.search(r'<conclusion>(.*?)</conclusion>', text, re.DOTALL)
    
    question = question_match.group(1).strip() if question_match else None
    thinking = think_match.group(1).strip() if think_match else None
    
    # Extract conclusion
    if conclusion_match:
        # Explicit conclusion tag
        conclusion = conclusion_match.group(1).strip()
    elif thinking:
        # No explicit conclusion tag - extract from thinking
        # For math problems, try to find the final numerical answer
        # Look for patterns like "= -338016." at the end
        final_line = thinking.strip().split('\n')[-1]
        
        # Try to extract final answer from last line
        # Pattern 1: "= answer" or "answer."
        answer_patterns = [
            r'=\s*([^\s\n]+)\.$',  # "= -338016."
            r'=\s*([^\s\n]+)$',    # "= 42"
            r'^([A-D])$',          # Single letter answer
        ]
        
        conclusion = None
        for pattern in answer_patterns:
            match = re.search(pattern, final_line.strip())
            if match:
                conclusion = match.group(1).strip()
                break
        
        # If still no conclusion, use the last substantial line
        if not conclusion:
            conclusion = final_line.strip()
    else:
        conclusion = None
    
    return question, thinking, conclusion


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model_and_tokenizer(config: Dict):
    """Initialize model and tokenizer"""
    logger.info("Loading model and tokenizer...")
    
    # Load tokenizer
    tokenizer = ReasoningLLMTokenizer('tokenizer/tokenizer.model')
    logger.info(f"Tokenizer loaded: vocab_size={len(tokenizer)}")
    
    # Create model config
    model_config = ReasoningLLMConfig(**config['model'])
    
    # Initialize model
    model = ReasoningLLMForCausalLM(model_config)
    logger.info(f"Model initialized: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    
    return model, tokenizer, model_config


def create_dataloader(config: Dict, tokenizer, is_eval: bool = False):
    """Create training or eval dataloader"""
    if is_eval:
        logger.info("Loading eval data...")
        data_path = config['data']['eval_data_path']
        max_examples = 100  # Fixed eval size
        skip_examples = 0
    else:
        logger.info("Loading training data...")
        data_path = config['data']['train_data_path']
        max_examples = config['data'].get('max_train_examples', None)
        skip_examples = config['data'].get('skip_train_examples', 0)
    
    max_tokens = config['data'].get('max_tokens_per_chunk', 512)
    
    dataset = RawTextDataset(
        data_path=config['data']['train_data_path'],
        tokenizer=tokenizer,
        max_examples=max_examples,
        skip_examples=skip_examples,
        max_tokens=max_tokens,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one question at a time for expert iteration
        shuffle=not is_eval,  # Shuffle train, not eval
        num_workers=0,  # No multiprocessing for simplicity
        pin_memory=False,  # Disable on Mac
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} examples")
    return dataloader


def generate_sample_response(model, tokenizer, question: str, device, max_length: int = 256):
    """Generate a sample response to see model output"""
    model.eval()
    
    # Prepare input
    prompt = f"Question: {question}\n\n"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    # Remove prompt
    response = generated_text[len(prompt):]
    
    return response


def compute_validation_loss(model, tokenizer, eval_dataloader, device, max_eval_steps: int = 20):
    """Compute validation loss on eval set"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            if i >= max_eval_steps:
                break
            
            text = batch['text'][0]
            
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    model.train()
    return avg_loss


def train_expert_iteration(config: Dict):
    """Main training loop with expert iteration"""
    
    # Setup
    model, tokenizer, model_config = create_model_and_tokenizer(config)
    dataloader = create_dataloader(config, tokenizer, is_eval=False)
    eval_dataloader = create_dataloader(config, tokenizer, is_eval=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Training on device: {device}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=(config['training']['adam_beta1'], config['training']['adam_beta2']),
        eps=config['training']['adam_epsilon'],
    )
    
    # Learning rate scheduler
    num_training_steps = config['training']['max_steps']
    scheduler = get_scheduler(
        config['training']['lr_scheduler'],
        optimizer=optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=num_training_steps,
    )
    
    # Load checkpoint if resuming
    start_step = 0
    checkpoint_path = config['checkpointing'].get('resume_from_checkpoint')
    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        training_state_path = Path(checkpoint_path) / 'training_state.pt'
        if training_state_path.exists():
            checkpoint = torch.load(training_state_path, map_location=device)
            start_step = checkpoint['step']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f"✓ Resumed from step {start_step}")
        else:
            logger.warning(f"No training_state.pt found in {checkpoint_path}, starting from step 0")
    
    # Expert iteration trainer
    ei_config = ExpertIterationConfig(
        num_samples=config['expert_iteration']['num_samples'],
        keep_top_k=config['expert_iteration']['keep_top_k'],
        temperature=config['expert_iteration']['temperature'],
        max_length=config['expert_iteration']['max_length'],
        verification_mode=config['expert_iteration']['verification_mode'],
        similarity_threshold=config['expert_iteration']['similarity_threshold'],
        min_quality_score=config['expert_iteration']['min_quality_score'],
        use_math_verifier=config['expert_iteration']['use_math_verifier'],
        log_samples=config['expert_iteration']['log_samples'],
        log_interval=config['expert_iteration']['log_interval'],
    )
    
    ei_trainer = ExpertIterationTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        config=ei_config,
        device=str(device),
    )
    
    # Training loop
    logger.info("="*80)
    if ei_config.num_samples == 0:
        logger.info("Starting SUPERVISED LEARNING (No Expert Iteration)")
        logger.info("Training directly on ground truth to learn format/grammar")
    else:
        logger.info("Starting Expert Iteration Training")
    logger.info("="*80)
    logger.info(f"Mode: {'SUPERVISED' if ei_config.num_samples == 0 else ei_config.verification_mode}")
    logger.info(f"Samples per question: {ei_config.num_samples}")
    logger.info(f"Keep top: {ei_config.keep_top_k}")
    logger.info(f"Max steps: {num_training_steps}")
    if ei_config.num_samples > 0:
        logger.info(f"Mode switch: When acceptance rate > {config['expert_iteration']['min_acceptance_rate_for_switch']:.0%} (after step {config['expert_iteration']['min_steps_before_switch']})")
    logger.info("="*80)
    
    global_step = start_step
    pbar = tqdm(total=num_training_steps, initial=start_step, desc="Training")
    
    data_iter = iter(dataloader)
    
    while global_step < num_training_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reset dataloader
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # Extract text from batch (batch_size=1, so just get first item)
        text = batch['text'][0]
        
        # Extract question and answer
        question, reference_thinking, ground_truth_conclusion = extract_question_and_answer(text)
        
        if question is None or ground_truth_conclusion is None:
            logger.warning(f"Step {global_step}: Failed to parse question/answer, skipping")
            continue
        
        # Prepare ground truth output for verification
        if reference_thinking:
            ground_truth_output = f"<think>{reference_thinking}</think><conclusion>{ground_truth_conclusion}</conclusion>"
        else:
            ground_truth_output = f"<conclusion>{ground_truth_conclusion}</conclusion>"
        
        # Expert iteration training step
        metrics = ei_trainer.train_step(
            question=f"Question: {question}",
            ground_truth_output=ground_truth_output,
            step=global_step,
        )
        
        # Update learning rate
        scheduler.step()
        
        # Check if we should switch to automatic mode (QUALITY-BASED)
        min_acceptance = config['expert_iteration']['min_acceptance_rate_for_switch']
        min_steps = config['expert_iteration']['min_steps_before_switch']
        
        stats = ei_trainer.get_stats()
        current_acceptance_rate = stats['acceptance_rate']
        
        # Switch only if: (1) minimum steps reached, (2) acceptance rate high enough, (3) not switched yet
        if (global_step >= min_steps and 
            current_acceptance_rate >= min_acceptance and 
            ei_trainer.verifier.mode == 'dataset'):
            logger.info("="*80)
            logger.info(f"🎯 QUALITY THRESHOLD REACHED at step {global_step}!")
            logger.info(f"Acceptance rate: {current_acceptance_rate:.1%} (threshold: {min_acceptance:.1%})")
            logger.info("SWITCHING TO AUTOMATIC VERIFICATION MODE")
            logger.info("Model can now explore creative reasoning paths!")
            logger.info("="*80)
            ei_trainer.switch_to_automatic_mode()
        
        # Logging
        if global_step % config['training']['logging_steps'] == 0:
            stats = ei_trainer.get_stats()
            if ei_config.num_samples == 0:
                # Supervised mode - simpler display
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'mode': 'SUP',
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                })
            else:
                # Expert iteration mode - full metrics
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'kept': f"{metrics['samples_kept']}/{metrics['samples_generated']}",
                    'score': f"{metrics['avg_quality_score']:.2f}",
                    'accept_rate': f"{stats['samples_accepted']/(stats['total_samples_generated']+1e-8):.1%}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                })
        
        # Save checkpoint
        if global_step % config['training']['save_steps'] == 0 and global_step > 0:
            save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, config)
        
        # Generate sample + validation loss every eval_steps
        eval_interval = config['training'].get('eval_steps', 50)
        if global_step % eval_interval == 0 and global_step > 0:
            logger.info("="*80)
            logger.info(f"📊 Evaluation at step {global_step}")
            
            # Validation loss
            val_loss = compute_validation_loss(model, tokenizer, eval_dataloader, device, max_eval_steps=20)
            logger.info(f"Validation Loss: {val_loss:.4f}")
            
            # Sample generation
            test_questions = [
                "What is 15 + 27?",
                "Solve for x: 2x + 5 = 13",
            ]
            for test_q in test_questions:
                sample_output = generate_sample_response(model, tokenizer, test_q, device, max_length=256)
                logger.info(f"\nQ: {test_q}")
                logger.info(f"A: {sample_output[:300]}..." if len(sample_output) > 300 else f"A: {sample_output}")
            
            logger.info("="*80)
            
            # Return to train mode
            model.train()
        
        global_step += 1
        pbar.update(1)
    
    pbar.close()
    
    # Final save
    logger.info("Training complete! Saving final checkpoint...")
    save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, config, is_final=True)
    
    # Print final statistics
    stats = ei_trainer.get_stats()
    logger.info("="*80)
    logger.info("FINAL STATISTICS")
    logger.info("="*80)
    logger.info(f"Training mode: {'SUPERVISED' if ei_config.num_samples == 0 else 'EXPERT ITERATION'}")
    if ei_config.num_samples > 0:
        logger.info(f"Total samples generated: {stats['total_samples_generated']}")
        logger.info(f"Samples accepted: {stats['samples_accepted']}")
        logger.info(f"Samples rejected: {stats['samples_rejected']}")
        logger.info(f"Acceptance rate: {stats['samples_accepted']/(stats['total_samples_generated']+1e-8):.1%}")
        logger.info(f"Average quality score: {stats['avg_quality_score']:.3f}")
    logger.info("="*80)


def save_checkpoint(model, tokenizer, optimizer, scheduler, step, config, is_final=False):
    """Save model checkpoint"""
    if is_final:
        checkpoint_dir = Path(config['checkpointing']['output_dir']) / "final"
    else:
        checkpoint_dir = Path(config['checkpointing']['output_dir']) / f"checkpoint-{step}"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving checkpoint to {checkpoint_dir}")
    
    # Save model
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    
    # Save training state
    torch.save({
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_dir / 'training_state.pt')
    
    logger.info(f"✓ Checkpoint saved")


def main():
    parser = argparse.ArgumentParser(description="Expert Iteration Training")
    parser.add_argument('--config', type=str, default='configs/test_config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Check expert iteration is enabled
    if not config.get('expert_iteration', {}).get('enabled', False):
        logger.error("Expert iteration is not enabled in config!")
        logger.error("Set expert_iteration.enabled: true in your config")
        sys.exit(1)
    
    # Train
    train_expert_iteration(config)


if __name__ == "__main__":
    main()
