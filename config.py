# 🔧 REPLACE the entire config.py with this streamlined version:

"""
Configuration settings for the Ultimate Resume Extractor - ENGLISH ONLY EDITION! 💅
"""

# --- Core Settings ---
RESUME_FOLDER = "merlion_resumes"
PROCESS_SUBFOLDERS = True  
OUTPUT_FILENAME_PREFIX = "extracted_resume_data"

# --- AI Model Selection (OPTIMIZED FOR ENGLISH!) ---
# 💅 Smaller, faster models for English-only processing!
FAST_MODEL = "llama3.2:1b"        # Quick validation checks
SMART_MODEL = "llama3.2:3b"       # Complex extractions
DEFAULT_MODEL = FAST_MODEL         # Use fast by default

MODEL_NAME = DEFAULT_MODEL
SUZUME_MODEL_NAME = SMART_MODEL    # For tough extractions only

# --- AI Feature Flags ---
USE_AI_EXTRACTION = True
AI_ONLY_FOR_FAILURES = True        # 🆕 Only call AI when regex fails!
AI_CONFIDENCE_THRESHOLD = 0.70     # Minimum confidence to accept regex result

# --- Performance Settings ---
AI_BATCH_SIZE = 10                 # Process multiple at once
AI_MAX_TOKENS = 300                # Reduced for faster responses
AI_TEMPERATURE = 0.05              # Very focused (0.0-1.0)

DEFAULT_BATCH_SIZE = 10
CHECKPOINT_FILE = "resume_checkpoint.json"

# --- Date Settings for Validation ---
MIN_AGE = 18                       # Minimum candidate age
MAX_AGE = 70                       # Maximum candidate age