#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmoRAG: Unified Emotion-Aware Audio Retrieval.
"""
import argparse, os, sys, json, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# METHOD 1: MLP-based Retrieval
# =============================================================================

def run_mlp_retrieval(args):
    print("--- Running MLP Retrieval Pipeline ---")
    try:
        import torch, torch.nn as nn, torch.nn.functional as F
    except ImportError: print("[ERROR] PyTorch is required for the 'mlp' method.", file=sys.stderr); sys.exit(1)
    try:
        import faiss
        HAS_FAISS = True
    except ImportError: print("[WARN] faiss not found, using numpy fallback.", file=sys.stderr); HAS_FAISS = False

    def encode_texts(texts, encoder_name):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(encoder_name)
            return model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to encode texts. Err={e}")

    def l2rows(M, eps=1e-9):
        n = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.maximum(n, eps)

    def pick_names(npz_obj):
        for key in ["filename", "fname", "stem"]:
            if key in npz_obj.files:
                suffix = ".wav" if key == "stem" else ""
                return np.array([os.path.basename(str(x)) + suffix for x in npz_obj[key]], dtype=object)
        N = next((v.shape[0] for v in npz_obj.values() if isinstance(v, np.ndarray) and v.ndim == 2), 0)
        return np.array([str(i) for i in range(N)], dtype=object)

    def build_index(Zy, names, index_dir, kind, try_gpu):
        os.makedirs(index_dir, exist_ok=True)
        np.save(os.path.join(index_dir, "Zy.npy"), Zy); np.save(os.path.join(index_dir, "names.npy"), names)
        if not HAS_FAISS: return None
        d = Zy.shape[1]
        if kind == "hnsw": idx = faiss.IndexHNSWFlat(d, 32); idx.hnsw.efConstruction = 80; idx.add(Zy.astype(np.float32))
        elif kind == "flat":
            base = faiss.IndexFlatIP(d)
            if try_gpu and faiss.get_num_gpus() > 0:
                gpu_idx = faiss.index_cpu_to_all_gpus(base); gpu_idx.add(Zy.astype(np.float32)); idx = faiss.index_gpu_to_cpu(gpu_idx)
            else: base.add(Zy.astype(np.float32)); idx = base
        else: raise ValueError(f"Unknown index kind: {kind}")
        faiss.write_index(idx, os.path.join(index_dir, f"faiss_{kind}_ip.index"))
        return idx

    def load_index_if_compatible(index_dir, Zy_shape, kind):
        if not HAS_FAISS: return None, None, None
        paths = {p: os.path.join(index_dir, f) for p, f in zip(["zy", "nm", "idx"], ["Zy.npy", "names.npy", f"faiss_{kind}_ip.index"])}
        if not all(os.path.exists(p) for p in paths.values()): return None, None, None
        Zy2 = np.load(paths["zy"]); nm2 = np.load(paths["nm"], allow_pickle=True)
        if Zy2.shape != Zy_shape: return None, None, None
        return faiss.read_index(paths["idx"]), Zy2, nm2

    def knn_search(Zq, Zy, topk, faiss_index=None):
        if faiss_index: return faiss_index.search(Zq.astype(np.float32), topk)
        sims = Zq @ Zy.T; I = np.argsort(-sims, axis=1)[:, :topk]
        return sims[np.arange(Zq.shape[0])[:, None], I], I

    class Head(nn.Module):
        def __init__(self, dx, rz): super().__init__(); self.net = nn.Sequential(nn.Linear(dx, 512), nn.ReLU(), nn.Linear(512, rz))
        def forward(self, x): return F.normalize(self.net(x), dim=-1)

    # --- Main Logic ---
    aud = np.load(args.audio_npz, allow_pickle=True)
    Y = aud.get("emo_emb", aud.get("audio_emb"))
    if Y is None: raise ValueError("audio_npz missing 'emo_emb' or 'audio_emb'.")
    names = pick_names(aud)

    if args.use_pca:
        print(f"[INFO] Using PCA projection mode from: {args.use_pca}")
        cca = np.load(args.use_pca, allow_pickle=True)
        Wy, y_mean = cca["Wy"], cca["y_mean"]
        Zy_mem = (Y - y_mean) @ Wy
    else:
        print("[INFO] Using direct alignment mode.")
        Zy_mem = Y
    
    Zy_mem = l2rows(Zy_mem)

    ckpt = torch.load(args.head_pt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    dx = next(v.shape[1] for k, v in sd.items() if k.endswith("weight") and v.ndim == 2)
    rz = ckpt.get("target_dim", Zy_mem.shape[1])
    model = Head(dx, rz); model.load_state_dict(sd, strict=True); model.eval()

    idx, Zy_cached, names_cached = load_index_if_compatible(args.index_dir, Zy_mem.shape, args.index_kind)
    if Zy_cached is None:
        print(f"[INFO] (Re)building index: kind={args.index_kind}, try_gpu={args.try_gpu}")
        idx = build_index(Zy_mem, names, args.index_dir, kind=args.index_kind, try_gpu=args.try_gpu)
        Zy_cached, names_cached = Zy_mem, names
    else:
        print("[INFO] Loaded cached index & embeddings.")

    queries, X, qnames = [], None, None
    if args.text: queries = [args.text.strip()]
    elif args.text_file: queries = [ln.strip() for ln in open(args.text_file, "r", encoding="utf-8") if ln.strip()]
    elif args.text_npz: X, qnames = np.load(args.text_npz, allow_pickle=True)["text_emb"], pick_names(np.load(args.text_npz, allow_pickle=True))
    if X is None: X, qnames = encode_texts(queries, args.encoder_model), np.array([f"q{i}" for i in range(len(queries))], dtype=object)

    with torch.no_grad(): Zx = model(torch.from_numpy(X)).cpu().numpy()

    D, I = knn_search(Zx, Zy_cached, topk=args.topk, faiss_index=idx)

    results = []
    for i in range(Zx.shape[0]):
        hits = [{"rank": j+1, "filename": str(names_cached[I[i, j]]), "path": os.path.join(args.wav_root, str(names_cached[I[i, j]])), "score": float(D[i, j])} for j in range(args.topk)]
        results.append({"query": str(qnames[i]), "text": (queries[i] if i < len(queries) else None), "hits": hits})

    for i, q in enumerate(results[:5]):
        print(f'\n=== Query[{i}] {q["text"] or q["query"]} ==='); [print(f'{h["rank"]:02d}. {h["path"]}  cos={h["score"]:.4f}') for h in q["hits"]]

    if args.out_json: 
        with open(args.out_json, "w", encoding="utf-8") as f: json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] saved results to: {args.out_json}")
    print("\n--- MLP Pipeline Finished ---")

# =============================================================================
# METHOD 2: VAD-based Retrieval
# =============================================================================

def run_vad_retrieval(args):
    print("--- Running VAD Retrieval Pipeline ---")
    from src.predict_vad import initialize_llm, get_vad_scores
    from src.find_audio import find_closest_audio
    try:
        initialize_llm()
    except ValueError as e:
        print(f"Fatal Error: {e}\nPlease ensure AWS_BEARER_TOKEN_BEDROCK is set.", file=sys.stderr); sys.exit(1)
    vad_scores = get_vad_scores(args.text)
    if not vad_scores or vad_scores.get('valence') is None: print("Fatal Error: Could not retrieve VAD scores.", file=sys.stderr); sys.exit(1)
    pV, pA = vad_scores['valence'], vad_scores['arousal']
    print(f"Predicted Scores -> Valence: {pV:.4f}, Arousal: {pA:.4f}")
    closest_files = find_closest_audio(pV, pA, args.vad_csv_path, topk=args.topk)
    if not closest_files: print("Fatal Error: Could not find any matching audio files.", file=sys.stderr); sys.exit(1)
    print(f'\n--- Top {len(closest_files)} Matches for "{args.text}" ---')
    results = []
    for i, row in enumerate(closest_files):
        print(f"{i+1:02d}. {row['FileName']} (V={row['EmoVal']:.4f}, A={row['EmoAct']:.4f}, Dist={row['distance']:.4f})")
        results.append({"rank": i+1, "filename": row['FileName'], "score": row['distance'], "valence": row['EmoVal'], "arousal": row['EmoAct']})
    if args.out_json: 
        with open(args.out_json, "w", encoding="utf-8") as f: json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] saved results to: {args.out_json}")

# =============================================================================
# MAIN COMMAND-LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="EmoRAG: Unified Emotion-Aware Audio Retrieval.", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="method", required=True, help="Retrieval method")

    p_mlp = subparsers.add_parser("mlp", help="Text -> MLP -> Audio Embedding Space")
    p_mlp_text_group = p_mlp.add_mutually_exclusive_group(required=True)
    p_mlp_text_group.add_argument("--text", help="Single text query")
    p_mlp_text_group.add_argument("--text_file", help="File with one query per line")
    p_mlp_text_group.add_argument("--text_npz", help="NPZ file with pre-computed text embeddings")
    p_mlp.add_argument("--head_pt", required=True, help="Path to trained MLP head model (.pt)")
    p_mlp.add_argument("--audio_npz", required=True, help="Path to pre-computed audio embeddings (.npz)")
    p_mlp.add_argument("--index_dir", required=True, help="Directory to store/load the FAISS index")
    p_mlp.add_argument("--use_pca", type=str, default=None, metavar="PATH_TO_CCA_NPZ", help="(Optional) Use PCA projection mode by providing path to a cca.npz file.")
    p_mlp.add_argument("--encoder_model", default="sentence-transformers/all-MiniLM-L6-v2")
    p_mlp.add_argument("--topk", type=int, default=10)
    p_mlp.add_argument("--index_kind", choices=["hnsw", "flat"], default="hnsw")
    p_mlp.add_argument("--try_gpu", action="store_true")
    p_mlp.add_argument("--force_rebuild", action="store_true")
    p_mlp.add_argument("--wav_root", default="", help="Optional root directory for audio paths")
    p_mlp.add_argument("--out_json", default="", help="Optional path to save results as JSON")

    p_vad = subparsers.add_parser("vad", help="Text -> LLM -> VAD Space -> Audio VAD Database")
    p_vad.add_argument("--text", required=True, help="The input text utterance to analyze.")
    p_vad.add_argument("--vad_csv_path", required=True, help="Path to the VAD metadata CSV for the audio library.")
    p_vad.add_argument("--topk", type=int, default=10)
    p_vad.add_argument("--out_json", default="", help="Optional path to save results as JSON")

    args = parser.parse_args()
    if args.method == "mlp": run_mlp_retrieval(args)
    elif args.method == "vad": run_vad_retrieval(args)

if __name__ == "__main__":
    main()