


import os
import sys
import logging
import warnings

# --- Environment and Path Setup ---



# --- Suppress Warnings ---
# Disable extensive logging from libraries to keep the output clean.
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore", module="tqdm")
# Set env var to avoid issues with some library initializations
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- Core sdialog Imports ---
try:
    import sdialog
    from sdialog import Dialog, Turn
    from sdialog.evaluation import LLMJudgeScore
except ImportError as e:
    print(f"Error: Failed to import a required library. The original error was: {e}")
    print("This might be a missing dependency in 'requirements.txt' or a problem with PYTHONPATH.")
    sys.exit(1)


# --- Core VAD Scoring Logic with Few-Shot Learning ---

def initialize_llm():
    """
    Initializes the LLM configuration.
    It expects the AWS Bedrock token to be set as an environment variable.
    """
    if not os.environ.get('AWS_BEARER_TOKEN_BEDROCK'):
        raise ValueError("Environment variable AWS_BEARER_TOKEN_BEDROCK is not set.")

    sdialog.config.set_llm("aws:anthropic.claude-3-5-sonnet-20240620-v1:0",
                           region_name="us-east-1")

class LLMValenceScore(LLMJudgeScore):
    """A reusable judge for scoring the Valence of an utterance."""
    def __init__(self, feedback: bool = True):
        prompt_template = (
            "You are an expert emotion analyst. Your task is to judge the valence (positivity/negativity) of an utterance. "
            "Provide a score from 1 (very negative) to 7 (very positive)."
            "\n--- Examples ---"
            "Utterance: 'This is the best day of my life!' -> Score: 7"
            "Utterance: 'I am so angry and disappointed.' -> Score: 1"
            "Utterance: 'It is an okay movie, not great but not terrible.' -> Score: 4"
            "\n--- Task ---"
            "Judge the valence of the following utterance and provide only the score."
            "Utterance: {{ dialog }}"
        )
        super().__init__(prompt_template, min_score=1, max_score=7)

class LLMArousalScore(LLMJudgeScore):
    """A reusable judge for scoring the Arousal of an utterance."""
    def __init__(self, feedback: bool = True):
        prompt_template = (
            "You are an expert emotion analyst. Your task is to judge the arousal (energy level) of an utterance. "
            "Provide a score from 1 (very calm, sleepy) to 7 (very active, energetic)."
            "\n--- Examples ---"
            "Utterance: 'I am so excited I can't sit still! Let's go!' -> Score: 7"
            "Utterance: 'I'm feeling very tired and peaceful.' -> Score: 1"
            "Utterance: 'I will just go to the store to buy some bread.' -> Score: 3"
            "\n--- Task ---"
            "Judge the arousal of the following utterance and provide only the score."
            "Utterance: {{ dialog }}"
        )
        super().__init__(prompt_template, min_score=1, max_score=7)

# --- Main Prediction Function ---

def get_vad_scores(utterance: str) -> dict:
    """
    Takes a single utterance as a string and returns its predicted valence and arousal scores.

    Args:
        utterance: The text to be analyzed.

    Returns:
        A dictionary with 'valence' and 'arousal' scores, or None if an error occurs.
    """
    # Initialize the judges
    valence_judge = LLMValenceScore()
    arousal_judge = LLMArousalScore()
    
    dialog = Dialog(turns=[Turn(text=utterance)])
    
    try:
        # The judges are callable and return the parsed score
        valence_score = valence_judge(dialog)
        arousal_score = arousal_judge(dialog)
        
        print(f"Raw scores - Valence: {valence_score}, Arousal: {arousal_score}") # Debug print

        # Ensure scores are numeric before returning
        return {
            "valence": float(valence_score) if valence_score is not None else None,
            "arousal": float(arousal_score) if arousal_score is not None else None
        }
    except Exception as e:
        print(f"An error occurred during VAD prediction: {e}")
        return {"valence": None, "arousal": None}

# This allows the file to be tested independently
if __name__ == '__main__':
    print("Testing predict_vad module...")
    try:
        # For direct testing, you must set the environment variable manually
        # For example, in your shell:
        # export AWS_BEARER_TOKEN_BEDROCK='your_token_here'
        initialize_llm()
        test_text = "I am so incredibly happy and excited about this!"
        scores = get_vad_scores(test_text)
        print(f"Utterance: \"{test_text}\"")
        if scores and scores['valence'] is not None:
            print(f"  - Predicted Valence: {scores['valence']:.2f}")
            print(f"  - Predicted Arousal: {scores['arousal']:.2f}")
        else:
            print("  - Failed to get scores.")
            
    except ValueError as e:
        print(f"Setup Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
