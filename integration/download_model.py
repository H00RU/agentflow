#!/usr/bin/env python3
"""
Download Qwen2.5-7B-Instruct model from Hugging Face
"""
import os
from huggingface_hub import snapshot_download

def download_qwen_model():
    """Download Qwen2.5-7B-Instruct model"""
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    local_dir = "/root/models/Qwen2.5-7B-Instruct"

    print(f"[Download] Starting download of {model_name}")
    print(f"[Download] Target directory: {local_dir}")
    print(f"[Download] This may take 10-20 minutes depending on network speed...")
    print("")

    try:
        # Download model
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print("")
        print(f"✅ [Download] Model downloaded successfully to {local_dir}")

        # Verify files
        files = os.listdir(local_dir)
        print(f"✅ [Download] Found {len(files)} files in model directory")

        # Check for key files
        key_files = ['config.json', 'tokenizer_config.json']
        for key_file in key_files:
            if key_file in files:
                print(f"✅ [Download] Found {key_file}")
            else:
                print(f"⚠️  [Download] Missing {key_file}")

        return True

    except Exception as e:
        print(f"❌ [Download] Error downloading model: {e}")
        return False

if __name__ == "__main__":
    success = download_qwen_model()
    exit(0 if success else 1)
