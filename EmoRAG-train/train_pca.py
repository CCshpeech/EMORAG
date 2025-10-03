#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmoRAG: PCA Projection Matrix Trainer

This script learns a PCA projection model from raw audio embeddings.
The output is an .npz file containing the projection matrix (Wy) and
mean vectors, compatible with the main training and retrieval scripts.
"""
import os
import argparse
import numpy as np
from sklearn.decomposition import PCA

def main(args):
    """Main function to train PCA and save the model."""
    print(f"[INFO] Loading audio embeddings from: {args.audio_npz}")
    try:
        train_data = np.load(args.audio_npz, allow_pickle=True)
        Y = train_data["emo_emb"].astype(np.float32)
    except Exception as e:
        print(f"[ERROR] Failed to load 'emo_emb' array from {args.audio_npz}. Error: {e}")
        return

    print(f"[INFO] Input audio embedding dimension: {Y.shape[1]}")
    print(f"[INFO] Target PCA dimension: {args.rank}")

    print("[INFO] Fitting PCA model... This may take a moment.")
    y_mean = Y.mean(axis=0, keepdims=True)
    Y_centered = Y - y_mean
    
    pca_model = PCA(n_components=args.rank, svd_solver="randomized", random_state=0).fit(Y_centered)
    Wy = pca_model.components_.T.astype(np.float32)

    # Create an output directory if it doesn't exist
    output_dir = os.path.dirname(args.out_npz)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the projection matrix and mean vectors
    np.savez_compressed(args.out_npz,
                        Wx=np.zeros((0, args.rank), np.float32),  # Placeholder for text matrix
                        Wy=Wy,
                        x_mean=np.zeros((0,), np.float32),  # Placeholder for text mean
                        y_mean=y_mean.ravel().astype(np.float32),
                        r=np.int32(args.rank))

    print(f"\n[SUCCESS] PCA projection model saved to: {args.out_npz}")
    print(f"[INFO] Output matrix 'Wy' shape: {Wy.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PCA projection matrix from raw audio embeddings.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--audio_npz", required=True, help="Path to the .npz file containing raw audio embeddings (e.g., train_audio_raw.npz).")
    parser.add_argument("--out_npz", required=True, help="Path to save the output .npz projection file.")
    parser.add_argument("--rank", type=int, default=256, help="The target dimension for the PCA projection.")

    args = parser.parse_args()
    main(args)
