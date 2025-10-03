#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmoRAG: Model Downloader

This script downloads the necessary model files for the MLP method.
"""
import os

try:
    import audeer
except ImportError:
    print("[ERROR] audeer is required. Please run 'pip install audeer'")
    exit(1)

# --- Configuration ---
MODELS = {
    "cca_RAW_train.npz": "https://huggingface.co/CindyChen19/z_head_L0.pt/resolve/main/cca_RAW_train.npz?download=true",
    "z_head_L0.pt": "https://huggingface.co/CindyChen19/z_head_L0.pt/resolve/main/z_head_L0.pt?download=true"
}

# --- Main Execution ---
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'models', 'mlp')
    audeer.mkdir(output_dir)

    print(f"[INFO] Models will be downloaded to: {output_dir}")

    for filename, url in MODELS.items():
        target_path = os.path.join(output_dir, filename)
        if os.path.exists(target_path):
            print(f"[INFO] Skipping {filename}, already exists.")
            continue
        
        print(f"[INFO] Downloading {filename}...")
        try:
            audeer.download_url(url, target_path, verbose=True)
            print(f"[INFO] Successfully downloaded {filename}.")
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")
            print("Please check the URL or your network connection.")

    print("\n--- Model Download Complete ---")
    print("You can now run the MLP method using the following paths:")
    print(f"  --cca_npz {os.path.join(output_dir, 'cca_RAW_train.npz')}")
    print(f"  --head_pt {os.path.join(output_dir, 'z_head_L0.pt')}")

if __name__ == "__main__":
    main()
