#!/usr/bin/env python3
"""
Clean up all local checkpoints immediately to free Mac storage.
Checkpoints are safely stored in HuggingFace Hub.
"""

import shutil
from pathlib import Path


def get_dir_size(path):
    """Get directory size in MB."""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except:
        pass
    return round(total / (1024 * 1024), 2)


def cleanup_checkpoints():
    print("=" * 60)
    print("Cleaning Up Local Checkpoints")
    print("=" * 60)
    print("\n⚠️  This will delete all local checkpoints to free Mac storage.")
    print("✅ Checkpoints are safely stored in HuggingFace Hub:")
    print("   https://huggingface.co/anaa01011/reasoning-llm-checkpoints")
    print()
    
    checkpoint_dirs = [
        "checkpoints/test_run",
        "checkpoints",
    ]
    
    total_freed = 0
    total_deleted = 0
    
    for checkpoint_base in checkpoint_dirs:
        base_path = Path(checkpoint_base)
        if not base_path.exists():
            continue
        
        # Find all checkpoint-* directories
        checkpoints = sorted(
            [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        )
        
        for checkpoint in checkpoints:
            size_mb = get_dir_size(checkpoint)
            try:
                shutil.rmtree(checkpoint)
                print(f"🗑️  Deleted: {checkpoint} (freed {size_mb} MB)")
                total_freed += size_mb
                total_deleted += 1
            except Exception as e:
                print(f"❌ Failed to delete {checkpoint}: {e}")
    
    print("\n" + "=" * 60)
    if total_deleted > 0:
        print(f"✓ Cleanup Complete!")
        print(f"  Deleted: {total_deleted} checkpoints")
        print(f"  Freed: {total_freed} MB (~{total_freed/1024:.2f} GB)")
    else:
        print("✓ No checkpoints found - Mac already clean!")
    print("=" * 60)
    print("\n💾 Your checkpoints are safe in HuggingFace Hub.")
    print("   Download anytime with:")
    print("   from huggingface_hub import snapshot_download")
    print("   snapshot_download(repo_id='anaa01011/reasoning-llm-checkpoints')")
    print("=" * 60)


if __name__ == "__main__":
    cleanup_checkpoints()
