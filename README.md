# EmoRAG: Emotion-Centric Audio Retrieval

This repository contains the official toolkit for the **EMORAG** project. Its core purpose is to serve as an upstream component for zero-shot Text-to-Speech (TTS) and voice cloning systems.

EMORAG's mission is to automatically select an emotionally appropriate audio prompt from a large voice library that matches the emotional content of an input text. This enables zero-shot TTS system to synthesize speech that is not only textually correct but also emotionally expressive, without manual emotion prompt selection.

This project provides two distinct, state-of-the-art methods for performing this cross-modal retrieval task.

## Features

- **Dual Retrieval Methods**: Choose between two powerful approaches:
  1.  **Direct MLP Alignment (Default)**: A simplified and powerful method that maps text embeddings directly to the raw audio embedding space.
  2.  **PCA-based MLP Alignment**: A legacy method that uses PCA to create a lower-dimensional shared space for text and audio.
  3.  **VAD-based Matching**: Uses large language models (LLMs) and direct audio analysis to match text and audio in the universal Valence-Arousal-Dominance (VAD) emotional space.
- **Bring Your Own Audio**: Comes with user-friendly command-line tools to automatically process your personal audio library, making it ready for retrieval.
- **Simplified Workflow**: A clean, unified interface for downloading models, preprocessing audio, and running retrieval.
- **Open & Extensible**: The project is structured to be easy to understand, use, and extend with new models or methods.

## Project Structure

- **`EmoRAG-infer/`**: The main directory containing the ready-to-use inference and preprocessing toolkit. **New users should start here.**
- **`EmoRAG-train/`**: Contains scripts to train your own MLP heads and PCA projection models. See the `README.md` inside for a detailed guide.

## Quick Start Guide

Follow these steps to get EmoRAG running with your own audio files.

### Step 1: Installation

First, clone the repository and install the required Python packages.

```bash
# Clone the project
git clone <your-repository-url> # Replace with your actual git repo URL

# Navigate to the inference directory
cd EmoRAG/EmoRAG-infer

# Install all dependencies
pip install -r requirements.txt
```

### Step 2: Download Models

The project uses several models. The VAD audio model (`w2v2-vad`) will be downloaded automatically on first use. For the MLP method, run the provided script to download the necessary components.

```bash
# From the EmoRAG-infer/ directory
python download_models.py
```

This will download the pre-trained MLP head model (`z_head_L0.pt`) and a PCA projection matrix (`cca_RAW_train.npz`) into the `EmoRAG-infer/models/mlp/` directory.

### Step 3: Prepare Your Audio Library

Process your own folder of `.wav` files using our simple preprocessing script.

**For the MLP Method:**

This command creates a searchable index from your audio files.

```bash
python preprocess.py mlp --audio_dir /path/to/your/wavs --output_dir ./my_mlp_index
```

**For the VAD Method:**

This command creates a CSV file with emotion scores for each audio file. **Note:** This requires the `AWS_BEARER_TOKEN_BEDROCK` environment variable to be set for making API calls.

```bash
export AWS_BEARER_TOKEN_BEDROCK='your_aws_bedrock_api_token'
python preprocess.py vad --audio_dir /path/to/your/wavs --output_csv ./my_audio_vad.csv
```

### Step 4: Run Retrieval!

Now you can query your audio library with text!

**Using the MLP Method (Direct Alignment - Default):**

This is the simplest and recommended way. It does not require the `--use_pca` flag.

```bash
python retrieve.py mlp \
    --text "I am so happy and excited I could jump for joy" \
    --head_pt ./models/mlp/z_head_L0.pt \
    --audio_npz ./my_mlp_index/audio_embeddings.npz \
    --index_dir ./my_mlp_index
```

**Using the MLP Method (with PCA):**

If you have a model trained with PCA, use the `--use_pca` flag.

```bash
python retrieve.py mlp \
    --text "I am so happy and excited I could jump for joy" \
    --head_pt ./models/mlp/z_head_L0.pt \
    --use_pca ./models/mlp/cca_RAW_train.npz \
    --audio_npz ./my_mlp_index/audio_embeddings.npz \
    --index_dir ./my_mlp_index
```

**Using the VAD Method:**

(Remember to set the `AWS_BEARER_TOKEN_BEDROCK` environment variable).

```bash
python retrieve.py vad \
    --text "A sad and rainy day, perfect for staying inside" \
    --vad_csv_path ./my_audio_vad.csv \
    --topk 5
```

## Advanced: Training

If you are interested in training your own models, please refer to the comprehensive guide in `EmoRAG-train/README.md`.

## License

This project is open-source. Please add a license file (e.g., MIT, Apache 2.0) to define how others can use your code.

```

```
