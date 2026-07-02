SEED = 42
DATASET_NAME = "artem9k/ai-text-detection-pile"

DETECTOR_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
TRAIN_SAMPLES_PER_CLASS = 2000
TEST_SIZE = 0.2

SECURITY_KEY = "Secure123"
WATERMARK_BITS = 16
AI_THRESHOLD = 0.50

DETECTOR_SAVE_PATH = "/kaggle/working/ai_detector_model"

PHRASE_TABLE = {
    "00": "In fact,",
    "01": "Additionally,",
    "10": "Moreover,",
    "11": "In summary,"
}

SEMANTIC_TABLE = {
    "00": "clearly",
    "01": "significantly",
    "10": "effectively",
    "11": "reliably"
}

SOURCE_TABLE = {
    "0001": "OpenAI / ChatGPT",
    "0010": "Google Gemini",
    "0011": "Anthropic Claude",
    "0100": "Meta LLaMA",
    "0101": "Student Custom Model",
    "0110": "Watermarked Research Generator",
    "0111": "Institutional AI System"
}

DEFAULT_SOURCE_ID = "0101"
