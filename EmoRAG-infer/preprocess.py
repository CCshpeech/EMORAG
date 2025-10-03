#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmoRAG: Unified Preprocessing Script for User Audio Libraries.
"""
import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd

# =============================================================================
# MLP PREPROCESSING IMPLEMENTATION
# =============================================================================

def preprocess_mlp(args):
    """Handler for the 'mlp' preprocessing method."""
    print("--- Starting MLP Preprocessing ---")
    
    try:
        from funasr import AutoModel
    except ImportError:
        print("[ERROR] funasr is required. Please run 'pip install funasr'", file=sys.stderr)
        sys.exit(1)

    try:
        import faiss
        HAS_FAISS = True
    except ImportError:
        print("[WARN] faiss not found, will skip index building.", file=sys.stderr)
        HAS_FAISS = False

    def build_faiss_index(Zy, index_dir, kind="hnsw"):
        if not HAS_FAISS: return
        print(f"[INFO] Building FAISS index of kind '{kind}'...")
        d = Zy.shape[1]
        idx = faiss.IndexHNSWFlat(d, 32); idx.hnsw.efConstruction = 80
        idx.add(Zy.astype(np.float32))
        index_path = os.path.join(index_dir, f"faiss_{kind}_ip.index")
        faiss.write_index(idx, index_path)
        print(f"[INFO] FAISS index saved to: {index_path}")

    print(f"Audio source directory: {args.audio_dir}")
    print(f"Output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    wav_files = glob.glob(os.path.join(args.audio_dir, "**", "*.wav"), recursive=True)
    if not wav_files: print(f"[ERROR] No .wav files found in {args.audio_dir}", file=sys.stderr); sys.exit(1)
    print(f"Found {len(wav_files)} .wav files to process.")

    print("[INFO] Loading emotion2vec model (this may take a while)...")
    # Corrected model path as per user's confirmation
    model = AutoModel(model="emotion2vec/emotion2vec_plus_base", hub="hf")
    print("[INFO] Model loaded.")

    embeddings, filenames = [], []
    for i, wav_path in enumerate(wav_files):
        try:
            rec = model.generate(wav_path, granularity="utterance", extract_embedding=True)
            vec = rec[0]["feats"] if isinstance(rec, list) and rec and "feats" in rec[0] else rec.get("feats")
            if vec is None: raise ValueError("Could not extract 'feats' from model output")
            embeddings.append(vec); filenames.append(os.path.basename(wav_path))
            if (i + 1) % 10 == 0 or (i + 1) == len(wav_files): print(f"Processed [{i+1}/{len(wav_files)}] {os.path.basename(wav_path)}")
        except Exception as e:
            print(f"[WARN] Failed to process {os.path.basename(wav_path)}: {e}", file=sys.stderr)

    if not embeddings: print("[ERROR] No embeddings extracted. Exiting.", file=sys.stderr); sys.exit(1)

    embeddings_np = np.array(embeddings, dtype=np.float32)
    output_npz_path = os.path.join(args.output_dir, "audio_embeddings.npz")
    np.savez_compressed(output_npz_path, emo_emb=embeddings_np, filename=np.array(filenames, dtype=object))
    print(f"[INFO] Embeddings and filenames saved to: {output_npz_path}")

    build_faiss_index(embeddings_np, args.output_dir)

    print("\n--- MLP Preprocessing Finished ---")
    print("You can now use the following paths with 'retrieve.py mlp':")
    print(f"  --audio_npz {output_npz_path}")
    print(f"  --index_dir {args.output_dir}")

# =============================================================================
# VAD PREPROCESSING IMPLEMENTATION
# =============================================================================

def preprocess_vad(args):
    """Handler for the 'vad' preprocessing method using a direct audio model."""
    print("--- Starting VAD Preprocessing ---")
    
    try:
        import audeer, audonnx, librosa
    except ImportError as e:
        print(f"[ERROR] Missing dependencies for VAD preprocessing: {e}. Please check requirements.txt", file=sys.stderr)
        sys.exit(1)

    model_url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_root = audeer.mkdir(os.path.join(script_dir, 'models', 'w2v2-vad'))
    
    if not os.path.exists(os.path.join(model_root, 'model.onnx')):
        print(f"[INFO] Downloading VAD model from {model_url}...")
        cache_root = audeer.mkdir(os.path.join(script_dir, 'models', 'cache'))
        archive_path = audeer.download_url(model_url, cache_root, verbose=True)
        audeer.extract_archive(archive_path, model_root)
        print("[INFO] Model downloaded and extracted.")
    else:
        print("[INFO] VAD model already exists.")

    print("[INFO] Loading VAD model...")
    model = audonnx.load(model_root)
    print("[INFO] VAD model loaded.")

    wav_files = glob.glob(os.path.join(args.audio_dir, "**", "*.wav"), recursive=True)
    if not wav_files: print(f"[ERROR] No .wav files found in {args.audio_dir}", file=sys.stderr); sys.exit(1)
    print(f"Found {len(wav_files)} .wav files to process.")

    results = []
    for i, wav_path in enumerate(wav_files):
        try:
            signal, sr = librosa.load(wav_path, sr=16000, mono=True)
            output = model(signal, sr)
            logits = output['logits'][0]
            results.append({
                'FileName': os.path.basename(wav_path),
                'EmoAct': logits[0], # Arousal
                'EmoDom': logits[1], # Dominance
                'EmoVal': logits[2]  # Valence
            })
            if (i + 1) % 10 == 0 or (i + 1) == len(wav_files): print(f"Processed [{i+1}/{len(wav_files)}] {os.path.basename(wav_path)}")
        except Exception as e:
            print(f"[WARN] Failed to process {os.path.basename(wav_path)}: {e}", file=sys.stderr)

    if not results: print("[ERROR] No VAD scores were generated. Exiting.", file=sys.stderr); sys.exit(1)

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"[INFO] VAD scores saved to: {args.output_csv}")

    print("\n--- VAD Preprocessing Finished ---")
    print("You can now use the following path with 'retrieve.py vad':")
    print(f"  --vad_csv_path {args.output_csv}")

# =============================================================================
# MAIN COMMAND-LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="EmoRAG: Preprocess your own audio library for retrieval.", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="method", required=True, help="Preprocessing method to use: 'mlp' or 'vad'")

    p_mlp = subparsers.add_parser("mlp", help="Generate embeddings and FAISS index for the MLP method.")
    p_mlp.add_argument("--audio_dir", required=True, help="Directory containing your .wav files.")
    p_mlp.add_argument("--output_dir", required=True, help="Directory to save the generated index files.")
    p_mlp.set_defaults(func=preprocess_mlp)

    p_vad = subparsers.add_parser("vad", help="Generate a CSV with VAD scores for the VAD method.")
    p_vad.add_argument("--audio_dir", required=True, help="Directory containing your .wav files.")
    p_vad.add_argument("--output_csv", required=True, help="Path to save the output .csv file.")
    p_vad.set_defaults(func=preprocess_vad)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()