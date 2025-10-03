# Training EmoRAG Models

This guide explains how to train the MLP projection head used by the `mlp` retrieval method in the `EmoRAG-infer` toolkit.

## Overview

This toolkit supports two training modes for the MLP head:

1.  **Direct Alignment (Default)**: This is the recommended and simplest method. It trains the MLP to map text embeddings directly to the raw, high-dimensional audio embedding space.
2.  **PCA Projection**: This is the legacy method. It first learns a PCA model to reduce the dimensionality of the audio embeddings, and then trains the MLP to map text embeddings to this lower-dimensional space.

## Prerequisites

Before you begin, you need a prepared dataset consisting of paired audio and text data. From this dataset, you must generate two files:

-   `train_audio_raw.npz`: An archive containing the raw audio embeddings for your training set. You can generate this using the `preprocess.py mlp` command in the `EmoRAG-infer` directory, which produces an `audio_embeddings.npz` file that you can rename and use.
-   `train_text_raw.npz`: An archive containing the raw text embeddings (e.g., from `all-MiniLM-L6-v2`) for the corresponding texts in your training set. The script `EmoRAG-train/data_prep/extract_text_embeds.py` can be used as a reference for this.

Place these two files in a working directory.

## Training Workflow

### Method 1: Direct Alignment (Default & Recommended)

This method trains the MLP head in a single step, without any dimensionality reduction.

**Action:**

Run the `train_l0.py` script, providing your prepared text and audio data, and specifying an output path for the trained model.

```bash
python scripts/l0/train_l0.py \
  --text_npz  /path/to/your/train_text_raw.npz \
  --audio_npz /path/to/your/train_audio_raw.npz \
  --out_pt    ./my_direct_model.pt \
  --bs 1024 \
  --lr 5e-4 \
  --steps 12000
```

This will create `my_direct_model.pt`. You can use this model directly with `retrieve.py` (without the `--use_pca` flag).

### Method 2: PCA Projection (Optional & Advanced)

This is a two-step process.

**Step 2.1: Train the PCA Projection Matrix**

First, use the `train_pca.py` script to learn the projection from your audio embeddings.

```bash
python train_pca.py \
    --audio_npz /path/to/your/train_audio_raw.npz \
    --out_npz   ./my_cca_pca.npz \
    --rank 256
```
This creates the `my_cca_pca.npz` file.

**Step 2.2: Train the MLP Head with PCA**

Now, run the main training script, but this time, provide the path to the PCA file you just created using the `--use_pca` flag.

```bash
python scripts/l0/train_l0.py \
  --text_npz  /path/to/your/train_text_raw.npz \
  --audio_npz /path/to/your/train_audio_raw.npz \
  --out_pt    ./my_pca_model.pt \
  --use_pca   ./my_cca_pca.npz
```

This will create `my_pca_model.pt`. To use this model during inference, you must provide the same `--use_pca ./my_cca_pca.npz` argument to `retrieve.py`.