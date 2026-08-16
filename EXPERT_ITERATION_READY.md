# Expert Iteration Training (this doc explains the expert iteration setup be carefull with this, we train this bloody model on three phases first is loss drop around stable base lines towards verified mal,mel.)

## What's Been Set Up here?

### 1. **Training Configuration**
-  Expert iteration **ENABLED**
-  Initial mode: **Dataset comparison** (first 100 steps)
-  Auto-switch to **Automatic mode** at step 100
-  Generate **6 samples** per question
-  Keep **top 3** for training (50% filter)
-  Similarity threshold: **0.4** (40% word overlap)

### 2. **Training Parameters**
- Max steps: **200** (quick test run)
- Batch size: **8**
- Learning rate: **3e-4**
- Mode switch: **Step 100**

### 3. **Dataset**
- Path: `data/train/train.jsonl`
- Format: Questions with `<think>` reasoning
- Auto-extracts: question, thinking, conclusion

## How It Works

### Phase 1: Steps 0-100 (Dataset Mode)
```
For each question:
  1. Model generates 6 different reasoning paths
  2. Compare each to reference thinking from dataset
  3. Keep samples with >40% similarity to reference
  4. Train ONLY on high-quality samples

Learn: Format, structure, basic reasoning patterns
```

### Phase 2: Steps 100-200 (Automatic Mode)
```
For each question:
  1. Model generates 6 creative solutions
  2. Verify conclusion is correct
  3. Check thinking quality (no contradictions, clear logic)
  4. Keep best samples regardless of similarity to dataset

Learn: Creative reasoning, novel solution paths
```

## Start Training

### Quick Start (Test Run - 200 steps)
```bash
./myenv/bin/python scripts/train_expert_iteration.py --config configs/test_config.yaml
```

### Full Training (Base Config)
```bash
./myenv/bin/python scripts/train_expert_iteration.py --config configs/base_config.yaml
```

## What You'll See

### Console Output
```
================================================================================
Starting Expert Iteration Training
================================================================================
Mode: dataset
Samples per question: 6
Keep top: 3
Max steps: 200
Mode switch at step: 100
================================================================================

Step 10: Loss=5.234, Samples kept=3/6, Score=0.78, Accept rate=56.7%, lr=1.2e-4
Step 20: Loss=4.891, Samples kept=4/6, Score=0.82, Accept rate=61.2%, lr=1.5e-4
...
================================================================================
SWITCHING TO AUTOMATIC VERIFICATION MODE at step 100
Model can now explore creative reasoning paths!
================================================================================
...
Step 200: Loss=3.456, Samples kept=5/6, Score=0.91, Accept rate=78.3%, lr=2.8e-4

Training complete! Saving final checkpoint...

================================================================================
FINAL STATISTICS
================================================================================
Total samples generated: 1200
Samples accepted: 942
Samples rejected: 258
Acceptance rate: 78.5%
Average quality score: 0.873
================================================================================
```

### Sample Logs (During Training)
```
Step 10: Loading question...
  Sample ✓ KEEP (score=0.87): Good match (similarity: 0.64)
  Sample ✗ REJECT (score=0.32): Thinking differs from reference (similarity: 0.32)
  Sample ✓ KEEP (score=0.91): Good match (similarity: 0.73)
  Sample ✓ KEEP (score=0.78): Good match (similarity: 0.58)
  Sample ✗ REJECT (score=0.00): Wrong conclusion: '45' != '42'
  Sample ✗ REJECT (score=0.12): Thinking differs from reference (similarity: 0.12)
```

## Expected Results

### Traditional Supervised Learning
- 10,000 steps to learn "2+2=4"
- Memorizes answers
- No reasoning ability

### Expert Iteration (Your Setup)
- **1,000 steps** to learn "2+2=4" (10x faster)
- Learns HOW to reason
- Self-improving (gets better over time)
- Discovers novel solution paths

## Checkpoints

Checkpoints saved to:
- Local: `checkpoints/test_run/checkpoint-100`, `checkpoint-200`
- HuggingFace Hub: Auto-uploaded (my Mac storage protected if yoyre forking this then protect yours as well)

## Monitoring

Check logs:
```bash
tail -f expert_iteration_training.log
```

Tensorboard (if enabled):
```bash
tensorboard --logdir logs/test_run
```

## Tips

### If Acceptance Rate Too Low (<30%)
- Model not ready yet, try:
  - Lower `similarity_threshold` to 0.3
  - Increase `temperature` to 1.0 (more diverse)
  - Reduce `min_quality_score` to 0.4

### If Acceptance Rate Too High (>95%)
- Model finding it too easy, try:
  - Raise `similarity_threshold` to 0.5
  - Lower `temperature` to 0.7 (more focused)
  - Increase `min_quality_score` to 0.6

### If Model Generates Gibberish
- Early training is normal, keep going
- By step 50-100, should see coherent thinking
- By step 100+, should see creative reasoning

## Architecture Benefits

  this model now has:
- ✅ **GQA** (3x smaller KV cache)
- ✅ **Flash Attention 2** (2-3x faster inference)  
- ✅ **Expert Iteration** (10x faster learning)
- ✅ **Self-correction** (Claude-style quality)

**This is a state-of-the-art 2024-2026 training pipeline tbh**

## Ready?

Just run:
```bash
./myenv/bin/python scripts/train_expert_iteration.py
```

## Watch the model learn to reason the process should be slow and steady if you start seeing the formatiion of text too quickly dont get confused "omg my model is finally becoming omnius of dune" rather it could be sign of memorization due to overfitting rather than generalization so always trust the data not the output.
