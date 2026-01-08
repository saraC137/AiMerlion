# 🔧 REPLACE the entire config.py with this streamlined version:

"""
config.py

This module centralizes all configuration settings for the AiMerlion Resume Extraction System.
It allows users to easily customize various parameters related to resume processing,
AI model interaction, PDF extraction, and OCR functionalities.

Key configurable sections include:
- Core Settings: Paths for resume folders and output files.
- AI Model Selection: Specifies the Ollama models to be used for different extraction tasks.
- AI Feature Flags: Controls the behavior of AI assistance in the extraction process.
- Performance Settings: Parameters for batch processing and AI response tuning.
- Date Settings: Age validation ranges for extracted dates of birth.
- PDF Extraction Settings: Options for using Marker PDF, pdfplumber, and their fallbacks.
- OCR Settings: Configuration for OCR engines (pytesseract, paddleocr, easyocr),
  languages, DPI, and hybrid extraction modes.
- Tesseract Configuration: Specific settings for the Tesseract OCR engine.
- Supported File Extensions: Defines the types of files the system can process.
"""

# --- Core Settings ---
RESUME_FOLDER = "merlion_resumes"
PROCESS_SUBFOLDERS = True  
OUTPUT_FILENAME_PREFIX = "extracted_resume_data"

# --- AI Model Selection (OPTIMIZED FOR ENGLISH!) ---
# 💅 Smaller, faster models for English-only processing!
FAST_MODEL = "llama3.1:8b"
SMART_MODEL = "llama3.1:8b"
DEFAULT_MODEL = FAST_MODEL
MODEL_NAME = DEFAULT_MODEL
SUZUME_MODEL_NAME = SMART_MODEL         # Use fast by default

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

# --- PDF Extraction Settings ---
USE_MARKER_PDF = True              # 🆕 Use Marker for better PDF extraction
MARKER_FALLBACK_TO_PDFPLUMBER = True  # Fall back to pdfplumber if Marker fails
MARKER_MAX_PAGES = None            # Process all pages (set to number to limit)

# --- OCR Settings ---
USE_OCR = True                     # Enable OCR for scanned/image PDFs
OCR_ENGINE = "pytesseract"         # Options: "pytesseract", "paddleocr", "easyocr"
OCR_LANGUAGE = "eng"               # Language for OCR (eng, chi_sim, jpn, etc.)
OCR_DPI = 200                      # DPI for PDF to image conversion (200 = best balance, same as test_ocr.py)
OCR_MIN_TEXT_THRESHOLD = 100       # Minimum chars before trying OCR
OCR_ALWAYS_RUN = False             # Set True to always run OCR and compare results
OCR_CONFIDENCE_THRESHOLD = 0.6     # Minimum confidence for PaddleOCR results

# --- Hybrid Extraction (for mixed PDFs with text + images) ---
HYBRID_EXTRACTION = True           # Combine text extraction + OCR for best results
HYBRID_MERGE_MODE = "combine"      # "combine" = merge both, "longest" = use longer result

# --- OCR Fallback Configuration (for vector/outline text PDFs) ---
OCR_CONFIG = {
    "dpi": 200,                    # DPI for PDF to image conversion (200 = good balance)
    "min_words_per_page": 20,      # Minimum words per page to consider extraction successful
    "ocr_timeout": 30,             # Timeout in seconds per page for OCR
    "min_text_length": 50,         # Minimum total characters to consider extraction valid
    "fallback_enabled": True,      # Enable automatic OCR fallback for failed extractions
}

# --- Tesseract Configuration ---
TESSERACT_CONFIG = "--oem 3 --psm 6"  # OEM 3 = default, PSM 6 = uniform block of text

# --- Supported File Extensions ---
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc"}