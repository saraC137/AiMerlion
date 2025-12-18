# Project Code Documentation

This document provides a high-level overview of the key Python files in the AiMerlion project.

## `main.py`

**Purpose:** This is the main entry point for the AiMerlion resume processing application. It provides a command-line interface (CLI) for users to interact with the system.

**Key Functionalities:**

*   **Interactive Menu:** Displays a user-friendly menu with options to:
    *   Start a fresh processing run on all resumes.
    *   Resume a previously interrupted run from a checkpoint.
    *   Select specific candidate folders to process.
    *   Run in a debug mode on a single folder.
    *   Identify and report on candidate folders that are missing resume files.
    *   List all candidates and their associated files.
*   **Orchestration:** Manages the overall workflow of reading candidate folders, calling the extractor, and generating reports.
*   **`UltimateResumeExtractor` Class:** This is the core class that handles the logic for extracting information from a candidate's resume files.
    *   It combines text from multiple resume files (PDF, DOCX).
    *   It uses a hybrid approach of regex and AI for data extraction.
    *   It leverages other modules like `ai_extractor`, `ai_validator`, and `marker_extractor`.
*   **Reporting:** After processing, it generates several CSV reports, including the main data extraction report, a report on fields that were missed, and a report on AI-assisted extractions.

## `config.py`

**Purpose:** This file acts as a central configuration hub for the entire application, allowing for easy tuning of various parameters without modifying the core logic.

**Key Settings:**

*   **File Paths:** Defines the location of the resume folder (`RESUME_FOLDER`) and the prefix for output files.
*   **AI Model Selection:** Specifies the names of the Ollama models to be used for different tasks (e.g., `FAST_MODEL`, `SMART_MODEL`).
*   **Feature Flags:** Contains boolean flags to enable or disable major features, such as `USE_AI_EXTRACTION` and `USE_MARKER_PDF`.
*   **Performance Tuning:** Includes settings like batch sizes, AI token limits, and temperature to control the performance and behavior of the AI models.
*   **Validation Parameters:** Defines business rules, such as the minimum and maximum age for candidates.

## `utils.py`

**Purpose:** A collection of utility functions and classes that support the main application logic, promoting code reuse and separation of concerns.

**Key Components:**

*   **CLI Helpers:** Functions like `display_menu` and `select_folders_to_process_enhanced` are responsible for rendering the interactive command-line interface.
*   **Checkpointing:** `save_checkpoint` and `load_checkpoint` functions allow the application to be stopped and resumed without losing progress.
*   **Data Standardization:** `standardize_phone_number` and `standardize_date` ensure that extracted data is converted into a consistent and clean format.
*   **Feedback & Learning Subsystem:** This is a major part of the `utils.py` file and includes several classes:
    *   `FeedbackLoopSystem`: Saves all extraction results and manual corrections, enabling the system to learn over time.
    *   `InteractiveCorrectionSystem`: Provides a UI for users to review and correct extracted data.
    *   `PatternLearningSystem`: Analyzes past corrections to suggest improvements to the extraction logic.
    *   `PerformanceMonitor`: Tracks key performance indicators (KPIs) of the extraction process.
*   **Text Extraction Helpers:** A set of functions (`extract_text_from_pdf`, `extract_text_from_docx`, etc.) for getting raw text from different file formats.

## `ai_extractor.py`

**Purpose:** This module is responsible for the heavy lifting of AI-based data extraction from resume text. It is designed to produce structured JSON output from unstructured text.

**Key Functionalities:**

*   **`AIExtractor` Class:**
    *   **Hybrid Extraction Strategy:** It employs a two-pass system:
        1.  **Header Extraction:** A quick pass on the beginning of the text to extract core contact details (name, email, phone).
        2.  **Deep Extraction:** A "manual-first" approach that uses robust regular expressions to extract skills, work experience, and education. It then uses the AI model in a targeted way to *categorize* the skills that the regex found.
    *   **Robust AI Interaction:**
        *   It constructs detailed, rule-based prompts to guide the LLM into producing clean, structured JSON.
        *   It includes a powerful `_fix_malformed_lists` function that acts as a "repair shop" for common LLM formatting errors (e.g., returning a string instead of a list).
        *   `_validate_deep_extraction` acts as a quality control check to detect when the AI has been "lazy" and just dumped text.
    *   **Ollama Integration:** Manages the communication with the Ollama backend.

## `ai_validator.py`

**Purpose:** This module acts as a specialized, lightweight "assistant" to the main extractor. It uses a smaller, faster LLM (like TinyLlama) to perform quick validation and correction tasks.

**Key Functionalities:**

*   **`AIValidator` Class:**
    *   **Name Validation:** Its `validate_name` function is a key feature. It asks the small LLM if a given piece of text is likely a person's name and returns a confidence score. This helps prevent job titles or other random words from being mistaken for names.
    *   **Targeted Extraction:** The `extract_from_messy_text` function is used as a fallback when the primary regex fails to find a critical field. It asks the LLM to find just one specific piece of information (e.g., "find only the phone number").
    *   **Text Repair:** Includes a `fix_vertical_text` function that uses the LLM to repair text that was formatted vertically in the original resume.
    *   **Data Validation:** Provides standard helper functions to validate the format of emails, phone numbers, and dates.

## `marker_extractor.py`

**Purpose:** This module provides a clean interface for extracting text from PDF files using the `marker-pdf` library, which is designed to convert PDFs into structured Markdown.

**Key Functionalities:**

*   **`MarkerExtractor` Class:**
    *   **Handles `marker-pdf` Initialization:** It properly loads the Marker models and creates a `PdfConverter` instance.
    *   **Error Handling:** If `marker-pdf` is not installed or fails to initialize, it gracefully disables itself, allowing the main application to fall back to other methods.
    *   **Singleton Pattern:** It uses the `get_marker_extractor` function to ensure that the heavy Marker models are loaded into memory only once, which is a critical performance optimization.
    *   **Extraction & Stats:** It extracts the PDF content as Markdown and provides a helper function to get statistics about the quality of the extraction (e.g., number of tables, lists detected).
