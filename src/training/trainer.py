"""
Training utilities and trainer class for Deep Reasoning LLM.
Includes mixed precision training, gradient accumulation, and distributed training support.
"""

import os
import math
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from transformers import get_scheduler
from tqdm import tqdm

from ..model import ReasoningLLMForCausalLM, ReasoningLLMConfig

try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class Trainer:
    """
    Trainer for Deep Reasoning LLM with all modern training features.
    """
    
    def __init__(
        self,
        model: ReasoningLLMForCausalLM,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            model: Model to train
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            config: Training configuration dictionary
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Default config
        default_config = {
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "warmup_steps": 2000,
            "max_steps": 100000,
            "gradient_accumulation_steps": 4,
            "eval_steps": 1000,
            "save_steps": 5000,
            "logging_steps": 100,
            "output_dir": "checkpoints",
            "log_dir": "logs",
            "fp16": False,
            "bf16": True,
            "lr_scheduler": "cosine",
            "min_lr": 3e-5,
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Setup device: prefer CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup mixed precision training
        # GradScaler only works on CUDA; MPS/CPU use autocast without scaler
        self.scaler = GradScaler("cuda") if (self.config["fp16"] and self.device.type == "cuda") else None
        # MPS: autocast is still experimental and causes CPU fallback for unsupported ops;
        # train in float32 on MPS for correctness and speed.
        if self.device.type == "mps":
            self.amp_dtype = torch.float32
            self.use_amp = False
        elif self.config["bf16"]:
            self.amp_dtype = torch.bfloat16
            self.use_amp = self.device.type == "cuda"  # BF16 autocast only meaningful on CUDA
        elif self.config["fp16"]:
            self.amp_dtype = torch.float16
            self.use_amp = self.device.type == "cuda"
        else:
            self.amp_dtype = torch.float32
            self.use_amp = False
        
        # Setup logging
        self.writer = SummaryWriter(log_dir=self.config["log_dir"])
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Create output directory
        Path(self.config["output_dir"]).mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("Trainer Initialized")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {'FP16' if self.config['fp16'] else 'BF16' if (self.config['bf16'] and self.use_amp) else 'FP32'}")
        print(f"Gradient Accumulation Steps: {self.config['gradient_accumulation_steps']}")
        print(f"Learning Rate: {self.config['learning_rate']}")
        print(f"Max Steps: {self.config['max_steps']}")
        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        print("=" * 60)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters that should and shouldn't use weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config["weight_decay"],
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config["learning_rate"],
            betas=(self.config["adam_beta1"], self.config["adam_beta2"]),
            eps=self.config["adam_epsilon"],
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        return get_scheduler(
            name=self.config["lr_scheduler"],
            optimizer=self.optimizer,
            num_warmup_steps=self.config["warmup_steps"],
            num_training_steps=self.config["max_steps"],
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single training step."""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass with mixed precision
        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / self.config["gradient_accumulation_steps"]
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config["gradient_accumulation_steps"]
    
    def optimizer_step(self):
        """Execute optimizer step with gradient clipping."""
        # Unscale gradients if using scaler
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config["max_grad_norm"]
        )
        
        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Scheduler step
        self.scheduler.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(**batch)
                loss = outputs.loss
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        
        self.model.train()
        
        return {
            "eval_loss": avg_loss,
            "perplexity": perplexity,
        }
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        # Clean up old checkpoints FIRST to save space
        self._cleanup_old_checkpoints()
        
        output_dir = Path(self.config["output_dir"]) / f"checkpoint-{step}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        
        # Save training state
        torch.save({
            "step": step,
            "epoch": self.epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
        }, output_dir / "training_state.pt")
        
        print(f"✓ Checkpoint saved to {output_dir}")
        
        # Upload to HuggingFace Hub if enabled
        if self.config.get("hub_upload_enabled", False):
            self._upload_to_hub(output_dir, step)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save Mac storage."""
        output_dir = Path(self.config["output_dir"])
        if not output_dir.exists():
            return
        
        # Get all checkpoint directories
        checkpoints = sorted(
            [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1])
        )
        
        save_total_limit = self.config.get("save_total_limit", 2)
        
        # Delete oldest checkpoints to make room for the new one about to be saved
        # Target: keep (save_total_limit - 1) existing ones so total = save_total_limit after save
        checkpoints_to_delete = checkpoints[:max(0, len(checkpoints) - (save_total_limit - 1))]
        
        for checkpoint in checkpoints_to_delete:
            try:
                size_mb = self._get_dir_size(checkpoint)
                shutil.rmtree(checkpoint)
                print(f"🗑️  Removed old checkpoint: {checkpoint.name} (freed {size_mb} MB)")
            except Exception as e:
                print(f"⚠️  Failed to remove {checkpoint}: {e}")
    
    def _upload_to_hub(self, checkpoint_dir: Path, step: int):
        """Upload checkpoint to HuggingFace Hub and optionally clean up."""
        if not HF_AVAILABLE:
            print("⚠️  HuggingFace Hub not available. Install: pip install huggingface_hub")
            self._force_cleanup_if_needed(checkpoint_dir)
            return
        
        hub_repo_id = self.config.get("hub_repo_id")
        hub_token = self.config.get("hub_token") or os.environ.get("HF_TOKEN")
        
        if not hub_repo_id or not hub_token:
            print("⚠️  HuggingFace Hub repo_id or token not configured")
            self._force_cleanup_if_needed(checkpoint_dir)
            return
        
        upload_success = False
        
        try:
            print(f"📤 Uploading checkpoint-{step} to HuggingFace Hub...")
            
            api = HfApi()
            
            # Create repo if it doesn't exist
            try:
                api.create_repo(repo_id=hub_repo_id, token=hub_token, exist_ok=True)
            except Exception as e:
                print(f"⚠️  Note: {e}")
            
            # Upload the checkpoint folder
            api.upload_folder(
                folder_path=str(checkpoint_dir),
                repo_id=hub_repo_id,
                token=hub_token,
                path_in_repo=f"checkpoint-{step}",
                commit_message=f"Upload checkpoint at step {step}"
            )
            
            print(f"✓ Checkpoint uploaded to https://huggingface.co/{hub_repo_id}")
            upload_success = True
            
        except Exception as e:
            print(f"❌ Failed to upload to HuggingFace Hub: {e}")
            upload_success = False
        
        finally:
            # Clean up based on configuration
            if self.config.get("cleanup_after_upload", False) and upload_success:
                self._cleanup_checkpoint(checkpoint_dir, "after successful upload")
            elif self.config.get("force_cleanup", False):
                # Force cleanup to save Mac storage even if upload failed
                self._cleanup_checkpoint(checkpoint_dir, "to save Mac storage (forced)")
    
    def _force_cleanup_if_needed(self, checkpoint_dir: Path):
        """Force cleanup if force_cleanup is enabled."""
        if self.config.get("force_cleanup", False):
            self._cleanup_checkpoint(checkpoint_dir, "to save Mac storage (forced - no upload)")
    
    def _cleanup_checkpoint(self, checkpoint_dir: Path, reason: str):
        """Clean up a specific checkpoint directory."""
        if not checkpoint_dir.exists():
            return
        
        try:
            size_mb = self._get_dir_size(checkpoint_dir)
            shutil.rmtree(checkpoint_dir)
            print(f"🗑️  Removed checkpoint: {checkpoint_dir.name} {reason} (freed {size_mb} MB)")
        except Exception as e:
            print(f"⚠️  Failed to remove checkpoint: {e}")
    
    def _get_dir_size(self, path: Path) -> float:
        """Get directory size in MB."""
        total = 0
        try:
            for entry in os.scandir(path):
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += sum(f.stat().st_size for f in Path(entry.path).rglob('*') if f.is_file())
        except:
            pass
        return round(total / (1024 * 1024), 2)
    
    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        
        self.model.train()
        
        running_loss = 0.0
        start_time = time.time()
        
        progress_bar = tqdm(total=self.config["max_steps"], desc="Training")
        
        while self.global_step < self.config["max_steps"]:
            for batch in self.train_dataloader:
                # Training step
                loss = self.train_step(batch)
                running_loss += loss
                
                # Optimizer step (if accumulated enough gradients)
                if (self.global_step + 1) % self.config["gradient_accumulation_steps"] == 0:
                    self.optimizer_step()
                
                self.global_step += 1
                progress_bar.update(1)
                
                # Logging
                if self.global_step % self.config["logging_steps"] == 0:
                    avg_loss = running_loss / self.config["logging_steps"]
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    steps_per_sec = self.config["logging_steps"] / elapsed
                    
                    self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                    self.writer.add_scalar("train/learning_rate", lr, self.global_step)
                    self.writer.add_scalar("train/steps_per_sec", steps_per_sec, self.global_step)
                    
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "steps/s": f"{steps_per_sec:.2f}",
                    })
                    
                    running_loss = 0.0
                    start_time = time.time()
                
                # Evaluation
                if self.global_step % self.config["eval_steps"] == 0:
                    eval_metrics = self.evaluate()
                    for key, value in eval_metrics.items():
                        self.writer.add_scalar(f"eval/{key}", value, self.global_step)
                    
                    if eval_metrics:
                        print(f"\nStep {self.global_step} - Eval Loss: {eval_metrics['eval_loss']:.4f}, "
                              f"Perplexity: {eval_metrics['perplexity']:.2f}")
                
                # Save checkpoint
                if self.global_step % self.config["save_steps"] == 0:
                    self.save_checkpoint(self.global_step)
                
                # Stop if reached max steps
                if self.global_step >= self.config["max_steps"]:
                    break
            
            self.epoch += 1
        
        # Final checkpoint
        self.save_checkpoint(self.global_step)
        
        progress_bar.close()
        self.writer.close()
        
        print("\n" + "=" * 60)
        print("Training Completed!")
        print("=" * 60)


if __name__ == "__main__":
    print("Training module")
    print("Use this via scripts/train.py")
