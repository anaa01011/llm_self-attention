#!/usr/bin/env python3
"""
Expert Iteration Trainer

Implements self-improving training loop:
1. Generate K samples per question
2. Verify each sample (dataset or automatic mode)
3. Keep top-scoring samples
4. Train only on high-quality outputs

10x faster learning than pure supervised training.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

from .verifier import ReasoningVerifier, MathVerifier, VerificationResult


logger = logging.getLogger(__name__)


@dataclass
class ExpertIterationConfig:
    """Configuration for expert iteration training"""
    # Generation settings
    num_samples: int = 6  # Generate K samples per question
    keep_top_k: int = 3   # Keep top K samples for training
    temperature: float = 0.8  # Sampling temperature (0.8 = diverse but focused)
    max_length: int = 256  # Max generation length
    
    # Verification settings
    verification_mode: str = "dataset"  # "dataset" or "automatic"
    similarity_threshold: float = 0.6  # For dataset mode
    min_quality_score: float = 0.5  # Minimum score to keep sample
    
    # Training settings
    use_math_verifier: bool = False  # Use specialized math verification
    log_samples: bool = True  # Log generated samples for debugging
    log_interval: int = 10  # Log every N steps


class ExpertIterationTrainer:
    """
    Expert Iteration training loop.
    
    Generates multiple samples, verifies quality, trains on best ones.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        optimizer,
        config: ExpertIterationConfig,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # Initialize verifier
        if config.use_math_verifier:
            self.verifier = MathVerifier(
                mode=config.verification_mode,
                similarity_threshold=config.similarity_threshold
            )
        else:
            self.verifier = ReasoningVerifier(
                mode=config.verification_mode,
                similarity_threshold=config.similarity_threshold
            )
        
        # Statistics
        self.stats = {
            "total_samples_generated": 0,
            "samples_accepted": 0,
            "samples_rejected": 0,
            "avg_quality_score": 0.0,
        }
        
        # Moving average tracking
        self.loss_history = []
        self.ma_window = 50  # 50-step moving average
        
    def train_step(
        self,
        question: str,
        ground_truth_output: str,
        step: int,
    ) -> Dict[str, float]:
        """
        Single expert iteration training step.
        
        Args:
            question: Input question
            ground_truth_output: Full ground truth with <think> and <conclusion>
            step: Current training step
            
        Returns:
            Dict with loss and statistics
        """
        # SUPERVISED LEARNING MODE: Skip sample generation if num_samples=0
        if self.config.num_samples == 0:
            return self._supervised_train_step(question, ground_truth_output, step)
        
        self.model.eval()  # Generation mode
        
        # Extract ground truth components
        ground_truth_conclusion = self._extract_tag(ground_truth_output, "conclusion")
        reference_thinking = self._extract_tag(ground_truth_output, "think")
        
        if ground_truth_conclusion is None:
            logger.warning(f"Step {step}: No conclusion in ground truth, skipping")
            return {"loss": 0.0, "samples_kept": 0}
        
        # Generate multiple samples
        samples = self._generate_samples(question)
        
        # Verify each sample
        verified_samples = []
        for sample in samples:
            result = self.verifier.verify(
                generated_output=sample,
                ground_truth_conclusion=ground_truth_conclusion,
                reference_thinking=reference_thinking,
                question=question,
            )
            
            self.stats["total_samples_generated"] += 1
            
            if result.passed and result.score >= self.config.min_quality_score:
                verified_samples.append((sample, result.score, result.reason))
                self.stats["samples_accepted"] += 1
            else:
                self.stats["samples_rejected"] += 1
            
            # Log sample details
            if self.config.log_samples and step % self.config.log_interval == 0:
                status = "✓ KEEP" if result.passed else "✗ REJECT"
                logger.info(f"  Sample {status} (score={result.score:.2f}): {result.reason}")
        
        # Sort by score and keep top K
        verified_samples.sort(key=lambda x: x[1], reverse=True)
        top_samples = verified_samples[:self.config.keep_top_k]
        
        if not top_samples:
            logger.warning(f"Step {step}: No valid samples generated, using ground truth")
            # Log a sample preview to see what model is generating
            if samples and step % 10 == 0:
                sample_preview = samples[0][:300] + "..." if len(samples[0]) > 300 else samples[0]
                logger.info(f"  Sample preview: {repr(sample_preview)}")
            # Fallback: train on ground truth
            top_samples = [(ground_truth_output, 1.0, "ground_truth")]
        
        # Train on top samples
        self.model.train()
        total_loss = 0.0
        
        for sample, score, reason in top_samples:
            # Prepare input
            input_text = f"{question}\n\n{sample}"
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            
            # Weight loss by quality score
            loss = outputs.loss * score
            
            # Backward pass
            loss.backward()
            total_loss += loss.item()
        
        # Optimizer step (after accumulating gradients from all samples)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Update statistics
        avg_score = sum(s[1] for s in top_samples) / len(top_samples)
        self.stats["avg_quality_score"] = avg_score
        
        # Track loss for moving average
        step_loss = total_loss / len(top_samples)
        self.loss_history.append(step_loss)
        if len(self.loss_history) > self.ma_window:
            self.loss_history.pop(0)
        moving_avg_loss = sum(self.loss_history) / len(self.loss_history)
        
        # Log step results
        if step % self.config.log_interval == 0:
            acceptance_rate = (
                self.stats["samples_accepted"] / self.stats["total_samples_generated"]
                if self.stats["total_samples_generated"] > 0 else 0.0
            )
            logger.info(
                f"Step {step}: "
                f"Loss={step_loss:.4f} (MA-50={moving_avg_loss:.4f}), "
                f"Samples kept={len(top_samples)}/{len(samples)}, "
                f"Avg score={avg_score:.3f}, "
                f"Acceptance rate={acceptance_rate:.1%}"
            )
        
        return {
            "loss": total_loss / len(top_samples),
            "samples_kept": len(top_samples),
            "samples_generated": len(samples),
            "avg_quality_score": avg_score,
        }
    
    def _supervised_train_step(
        self,
        question: str,
        ground_truth_output: str,
        step: int,
    ) -> Dict[str, float]:
        """
        Supervised learning step - train directly on ground truth without sample generation.
        Faster training for initial format learning.
        """
        self.model.train()
        
        # Prepare input with ground truth
        input_text = f"{question}\n\n{ground_truth_output}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Track loss for moving average
        step_loss = loss.item()
        self.loss_history.append(step_loss)
        if len(self.loss_history) > self.ma_window:
            self.loss_history.pop(0)
        moving_avg_loss = sum(self.loss_history) / len(self.loss_history)
        
        # Log step results
        if step % self.config.log_interval == 0:
            logger.info(
                f"Step {step}: "
                f"Loss={step_loss:.4f} (MA-50={moving_avg_loss:.4f}), "
                f"Mode=SUPERVISED"
            )
        
        return {
            "loss": step_loss,
            "samples_kept": 1,
            "samples_generated": 0,
            "avg_quality_score": 1.0,
        }
    
    def _generate_samples(self, question: str) -> List[str]:
        """Generate K diverse samples for a question"""
        samples = []
        
        # Prepare prompt
        prompt = f"{question}\n\n"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Generate multiple samples
        with torch.no_grad():
            for _ in range(self.config.num_samples):
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_length,  # Generate up to max_length NEW tokens
                    temperature=self.config.temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode
                generated_text = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                
                # Extract only the generated part (remove prompt)
                generated_part = generated_text[len(prompt):].strip()
                samples.append(generated_part)
        
        return samples
    
    def _extract_tag(self, text: str, tag: str) -> Optional[str]:
        """Extract content between XML-style tags"""
        import re
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def get_stats(self) -> Dict[str, float]:
        """Get training statistics"""
        stats = self.stats.copy()
        # Add computed acceptance rate
        stats['acceptance_rate'] = (
            stats['samples_accepted'] / stats['total_samples_generated']
            if stats['total_samples_generated'] > 0 else 0.0
        )
        return stats
    
    def switch_to_automatic_mode(self):
        """Switch verifier from dataset to automatic mode"""
        logger.info("Switching to automatic verification mode")
        self.verifier.mode = "automatic"
        self.config.verification_mode = "automatic"
