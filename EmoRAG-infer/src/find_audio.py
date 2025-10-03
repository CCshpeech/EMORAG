import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np

def find_closest_audio(predicted_valence: float, predicted_arousal: float, csv_path: str, topk: int = 1) -> list[dict] | None:
    """
    Finds the top-k audio files in the CSV with the closest Valence and Arousal scores.

    Args:
        predicted_valence: The predicted valence score.
        predicted_arousal: The predicted arousal score.
        csv_path: The absolute path to the metadata CSV file.
        topk: The number of closest matches to return.

    Returns:
        A list of dictionaries, where each dictionary represents a matched audio file,
        sorted by distance. Returns None if an error occurs.
    """
    if predicted_valence is None or predicted_arousal is None:
        print("Error: Invalid predicted VAD scores (None).")
        return None

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.", file=sys.stderr)
        return None

    required_cols = ['EmoVal', 'EmoAct', 'FileName']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain the columns: {required_cols}", file=sys.stderr)
        return None

    df.dropna(subset=['EmoVal', 'EmoAct'], inplace=True)
    df['EmoVal'] = pd.to_numeric(df['EmoVal'], errors='coerce')
    df['EmoAct'] = pd.to_numeric(df['EmoAct'], errors='coerce')
    df.dropna(subset=['EmoVal', 'EmoAct'], inplace=True)

    if df.empty:
        print("Error: No valid data rows in the CSV after cleaning.", file=sys.stderr)
        return None

    ground_truth_scores = df[['EmoVal', 'EmoAct']].values
    predicted_score = np.array([[predicted_valence, predicted_arousal]])

    # Calculate Euclidean distances
    distances = cdist(predicted_score, ground_truth_scores, 'euclidean').flatten()

    # Get the indices of the `topk` smallest distances
    topk_indices = np.argsort(distances)[:topk]

    # Get the corresponding rows from the dataframe
    closest_matches = df.iloc[topk_indices].copy()
    
    # Add the distance to the results
    closest_matches['distance'] = distances[topk_indices]

    return closest_matches.to_dict('records')

# This allows the file to be tested independently
if __name__ == '__main__':
    print("Testing find_audio module with topk support...")
    
    dummy_data = {
        'FileName': ['audio1.wav', 'audio2.wav', 'audio3.wav', 'audio4.wav'],
        'EmoVal': [1.0, 7.0, 4.0, 6.5],
        'EmoAct': [1.0, 7.0, 4.0, 6.8]
    }
    dummy_csv_path = './dummy_msp_info.csv'
    pd.DataFrame(dummy_data).to_csv(dummy_csv_path, index=False)

    print("\n--- Test Case 1: Find top 2 for Happy/Excited ---")
    pred_v, pred_a = 6.8, 6.9
    top_k = 2
    print(f"Predicted V/A: ({pred_v}, {pred_a}), Top K: {top_k}")
    closest_files = find_closest_audio(pred_v, pred_a, dummy_csv_path, topk=top_k)
    
    if closest_files:
        for i, f in enumerate(closest_files):
            print(f"  {i+1}. {f['FileName']} (Dist: {f['distance']:.4f})")
    # Expected: audio2.wav and audio4.wav in some order

    print("\n--- Test Case 2: Find top 1 for Sad/Calm ---")
    pred_v, pred_a = 1.2, 1.5
    top_k = 1
    print(f"Predicted V/A: ({pred_v}, {pred_a}), Top K: {top_k}")
    closest_files = find_closest_audio(pred_v, pred_a, dummy_csv_path, topk=top_k)
    if closest_files:
        for i, f in enumerate(closest_files):
            print(f"  {i+1}. {f['FileName']} (Dist: {f['distance']:.4f})")
    # Expected: audio1.wav

    import os
    os.remove(dummy_csv_path)