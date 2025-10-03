#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def main(args):
    """Main function to extract text embeddings."""
    print(f"[INFO] Loading input CSV from: {args.input_csv}")
    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {args.input_csv}")
        return

    if args.transcript_column not in df.columns:
        print(f"[ERROR] Transcript column '{args.transcript_column}' not found in the CSV.")
        print(f"Available columns are: {list(df.columns)}")
        return

    # Clean and filter transcripts
    df[args.transcript_column] = df[args.transcript_column].astype(str).str.strip()
    df.dropna(subset=[args.transcript_column], inplace=True)
    df = df[df[args.transcript_column].str.len() > 0].copy()

    if df.empty:
        print("[ERROR] No valid text found in the specified column.")
        return

    print(f"[INFO] Found {len(df)} valid transcripts to encode.")

    print(f"[INFO] Loading Sentence Transformer model: {args.model_name_or_path}")
    model = SentenceTransformer(args.model_name_or_path, device=args.device)

    # Prepare text list for encoding
    texts = [" ".join(t.split()) for t in df[args.transcript_column].tolist()]

    print(f"[INFO] Encoding texts with batch size {args.batch_size}...")
    emb = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)
    emb = emb.astype(np.float32)

    # Prepare data for saving
    save_data = {"text_emb": emb}
    for col in ["FileName", "Split_Set"]:
        if col in df.columns:
            save_data[col.lower()] = df[col].values
    save_data["transcript"] = df[args.transcript_column].values

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_npz)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(args.output_npz, **save_data)
    print(f"\n[SUCCESS] Saved embeddings to: {args.output_npz}")
    print(f"[INFO] Output embedding shape: {emb.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract text embeddings from a CSV file using Sentence Transformers.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file (e.g., MSP-info.csv).")
    parser.add_argument("--transcript_column", required=True, help="Name of the column containing the text transcripts.")
    parser.add_argument("--output_npz", required=True, help="Path to save the output .npz file.")
    parser.add_argument("--model_name_or_path", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Name of the Sentence Transformer model or path to a local model directory.")
    parser.add_argument("--device", default=None, help="Device to use for computation (e.g., 'cuda', 'cpu'). Defaults to auto-detection.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for encoding.")

    args = parser.parse_args()
    main(args)