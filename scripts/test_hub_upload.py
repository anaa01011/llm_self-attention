#!/usr/bin/env python3
"""
Test HuggingFace Hub checkpoint upload
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import HfApi

def test_hub_upload():
    print("=" * 60)
    print("Testing HuggingFace Hub Upload")
    print("=" * 60)
    
    # Your credentials from config
    hub_token = "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"
    hub_repo_id = "anaa01011/reasoning-llm-checkpoints"
    
    # Test checkpoint path (using existing checkpoint)
    checkpoint_path = "checkpoints/test_run/checkpoint-200"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please run training first to create checkpoints")
        return
    
    try:
        print(f"\n1. Authenticating with HuggingFace...")
        api = HfApi()
        
        print(f"2. Creating/checking repo: {hub_repo_id}...")
        api.create_repo(
            repo_id=hub_repo_id,
            token=hub_token,
            exist_ok=True,
            private=False  # Set to True if you want private repo
        )
        print(f"   ✓ Repo ready: https://huggingface.co/{hub_repo_id}")
        
        print(f"\n3. Uploading checkpoint...")
        api.upload_folder(
            folder_path=checkpoint_path,
            repo_id=hub_repo_id,
            token=hub_token,
            path_in_repo="checkpoint-200",
            commit_message="Test upload from Mac"
        )
        
        print("\n" + "=" * 60)
        print("✓ Upload Successful!")
        print("=" * 60)
        print(f"\nYour checkpoint is now available at:")
        print(f"https://huggingface.co/{hub_repo_id}/tree/main/checkpoint-200")
        print("\nYou can download it later with:")
        print(f"from huggingface_hub import snapshot_download")
        print(f"snapshot_download(repo_id='{hub_repo_id}', local_dir='./downloaded_checkpoint')")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your token is valid: https://huggingface.co/settings/tokens")
        print("2. Make sure you're logged in: huggingface-cli login")
        print("3. Check internet connection")


if __name__ == "__main__":
    test_hub_upload()
