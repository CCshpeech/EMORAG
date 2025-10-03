#!/usr/bin/env python3
import os, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def l2(x, eps=1e-9): return x / (x.norm(dim=-1, keepdim=True) + eps)

class Head(nn.Module):
    def __init__(self, dx, rz):
        super().__init__()
        # A simple MLP head
        self.net = nn.Sequential(
            nn.Linear(dx, 512), # Increased intermediate layer size
            nn.ReLU(),
            nn.Linear(512, rz)
        )
    def forward(self, x): return l2(self.net(x))

def info_nce(zq, zk, t=0.07):
    """InfoNCE contrastive loss."""
    logits = (zq @ zk.t()) / t
    target = torch.arange(zq.size(0), device=zq.device)
    return nn.CrossEntropyLoss()(logits, target)

def main(a):
    # Load text and audio embeddings
    tx = np.load(a.text_npz, allow_pickle=True)
    X = tx["text_emb"].astype(np.float32)
    ax = np.load(a.audio_npz, allow_pickle=True)
    Y = ax["emo_emb"].astype(np.float32)

    # --- Determine training mode: PCA or Direct ---
    if a.use_pca:
        print(f"[INFO] Using PCA projection mode from: {a.use_pca}")
        cca = np.load(a.use_pca, allow_pickle=True)
        Wy = cca["Wy"]
        y_mean = cca["y_mean"]
        r = int(cca["r"])
        # Project audio embeddings to the PCA subspace
        Zy = (Y - y_mean[None, :]) @ Wy
        print(f"[INFO] Audio embeddings projected from {Y.shape[1]} -> {Zy.shape[1]} dims.")
    else:
        print("[INFO] Using direct alignment mode.")
        # Target dimension is the original audio embedding dimension
        r = Y.shape[1]
        # Target embeddings are the raw audio embeddings
        Zy = Y
        print(f"[INFO] Target alignment dimension: {r}")

    # L2-normalize the target audio embeddings
    Zy = (Zy / (np.linalg.norm(Zy, axis=1, keepdims=True) + 1e-9)).astype(np.float32)

    # --- DataLoader and Model Setup ---
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Zy))
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Training on device: {device}")
    model = Head(X.shape[1], r).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    # --- Training Loop ---
    step = 0
    it = iter(dl)
    model.train()
    print("[INFO] Starting training...")
    while step < a.steps:
        try:
            xb, zy = next(it)
        except StopIteration:
            it = iter(dl)
            continue
        
        xb = xb.to(device, non_blocking=True)
        zy = zy.to(device, non_blocking=True)

        zq = model(xb)
        loss = info_nce(zq, zy, t=a.temp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.clip > 0: nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()

        step += 1
        if step % a.log_every == 0:
            with torch.no_grad():
                diag_cos = (zq * zy).sum(dim=1).mean().item()
            print(f"step {step}/{a.steps} | loss {loss.item():.4f} | diag-cos {diag_cos:.4f}", flush=True)

    # --- Save Model ---
    output_dir = os.path.dirname(a.out_pt)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save metadata along with the model state
    save_obj = {"state_dict": model.state_dict(), "target_dim": r}
    if a.use_pca:
        save_obj["pca_projection"] = True

    torch.save(save_obj, a.out_pt)
    print(f"[SUCCESS] Model saved to: {a.out_pt}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train an MLP head to map text embeddings to an audio embedding space.")
    # Required arguments
    ap.add_argument("--text_npz", required=True, help="Path to the .npz file with text embeddings.")
    ap.add_argument("--audio_npz", required=True, help="Path to the .npz file with audio embeddings.")
    ap.add_argument("--out_pt", required=True, help="Path to save the output PyTorch model file.")

    # Optional PCA mode
    ap.add_argument("--use_pca", type=str, default=None, metavar="PATH_TO_CCA_NPZ",
                        help="(Optional) Path to a pre-trained PCA/CCA projection file. If provided, enables PCA mode. Otherwise, direct alignment is used.")

    # Hyperparameters
    ap.add_argument("--bs", type=int, default=1024, help="Batch size.")
    ap.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    ap.add_argument("--temp", type=float, default=0.07, help="InfoNCE temperature parameter.")
    ap.add_argument("--steps", type=int, default=12000, help="Total training steps.")
    ap.add_argument("--log_every", type=int, default=200, help="Log progress every N steps.")
    ap.add_argument("--clip", type=float, default=1.0, help="Gradient clipping value (0 to disable).")
    
    args = ap.parse_args()
    main(args)