# AiMerlion: Technical File Documentation & Architecture

This document provides a detailed breakdown of every file in the AiMerlion project, explaining their individual roles and how they connect to form a cohesive resume processing ecosystem.

---

## 1. Orchestration & UI (The Stage Managers)

These files control the high-level flow of the application and how the user interacts with it.

*   **`main.py`**: The heart of the project. It provides the interactive CLI menu and orchestrates the entire "Ultimate Resume Extractor" workflow. It coordinates between text extraction, AI processing, database storage, and reporting.
*   **`config.py`**: The central brain for settings. It defines model names (Ollama), file paths, confidence thresholds, and feature flags (e.g., `USE_AI_EXTRACTION`).
*   **`utils.py`**: A massive toolbox containing CLI helpers (menus), data standardization (phone/date cleaners), and a sophisticated **Feedback & Learning System** that allows the system to improve from manual corrections.
*   **`mainONE.py`**: A secondary entry point, likely used for specific single-file processing tasks or as a simplified legacy version of the main pipeline.

---

## 2. Extraction Core (The Heavy Lifters)

These files are responsible for turning messy documents into structured data.

*   **`ai_extractor.py`**: Implements the hybrid extraction strategy. It uses **Regex-first** for structured sections (Experience/Education) and **AI-first** for contact headers. It handles prompt engineering and LLM response parsing.
*   **`document_parser.py`**: Handles the dirty work of reading PDFs and DOCX files. It includes a robust **OCR fallback** (using PaddleOCR) for scanned images or non-selectable text.
*   **`ai_validator.py`**: A "sanity checker" module. It uses smaller LLMs to validate names, repair messy text, and perform targeted extraction when the primary logic fails.
*   **`marker_extractor.py`**: An advanced PDF-to-Markdown converter using the `marker-pdf` library. It's used when high-fidelity structural preservation is needed.
*   **`ner_schema.py`**: Defines the JSON schemas and entity types for Named Entity Recognition (NER), ensuring the AI output is consistent.
*   **`optimized_prompts.py`**: A dedicated storage file for refined, high-performance prompts used to talk to the LLMs.

---

## 3. Data Management & Storage (The Vault)

These files handle the persistence and health of your data.

*   **`db_manager.py`**: Manages a three-layer SQLite database: `raw_extractions` (raw text), `structured_extractions` (the glam results), and `extraction_log` (the history of how every field was found).
*   **`db_diagnostic.py`**: A utility script to check the database for corruption, missing indices, or schema version mismatches.
*   **`fix_cache.py`**: A maintenance script used to clear or repair the local AI model caches and temporary processing files.

---

## 4. Machine Learning & Intelligence (The Brains)

Beyond simple extraction, these files provide "smart" features and training capabilities.

*   **`ml_engine.py`**: An advanced module using Scikit-Learn to provide:
    *   **Confidence Scoring:** Predicts how likely an extraction is correct.
    *   **Anomaly Detection:** Flags weird data (like a 50-digit phone number).
    *   **Prioritized Review:** Ranks resumes so humans check the "riskiest" ones first.
*   **`finetune_model.py` & `train_ner.py`**: Scripts used to train or fine-tune local models on your specific resume dataset for higher accuracy.
*   **`merge_lora.py`**: Handles merging LoRA (Low-Rank Adaptation) weights into base models after fine-tuning.
*   **`prompt_optimizer.py`**: An experimental tool that uses AI to "self-correct" and improve the prompts in `optimized_prompts.py` based on extraction failures.

---

## 5. Classification & Evaluation (The Analysts)

Tools for measuring performance and categorizing the dataset.

*   **`resume_classifier.py`**: Scans resumes to determine if they are in English, Japanese, or both, which helps the system choose the right language model.
*   **`evaluator.py`**: The "Grading Tool." It compares the system's output against a "Gold Standard" (ground truth) dataset to generate accuracy reports.
*   **`resume_exporter.py`**: Converts the database results into various user-friendly formats like CSV, Excel, or JSON.
*   **`compare_extraction.py`**: A diagnostic tool that allows you to see the "Before vs. After" of an extraction when you change your code or prompts.

---

## 6. Dashboards & Data Preparation

*   **`classification_dashboard.py` & `review_dashboard.py`**: (Likely) Plotly/Streamlit/Flask-based interfaces for visually reviewing classification results and correcting data.
*   **`labeling_tool.py` & `annotation_tool.py`**: Interactive CLI or GUI tools for humans to label raw resume text, creating the training data needed for the ML engine.
*   **`converter.py`**: A utility for converting between different labeling formats (e.g., LabelStudio to NER).
*   **`extract_raw_text.py`**: A simple utility to dump the raw text of all resumes into text files for manual inspection.

---

## 7. Diagnostics & Testing (The Inspectors)

*   **`diagnose.py` / `diagnose_ai.py` / `diagnose_pdf.py`**: Specialized scripts to test if your AI connection is working or if your PDF library is behaving correctly.
*   **`pdf_inspector.py`**: A technical tool that looks "under the hood" of a PDF to see how it's encoded.
*   **`test_*.py`**: A comprehensive suite of tests (over 10 files) that verify everything from the OCR accuracy to the database integrity.

---

## 🔄 System Interconnections

1.  **Initialization:** `main.py` loads `config.py` and initializes `DatabaseManager` and `MLEngine`.
2.  **Input:** `main.py` uses `document_parser.py` to convert files in `merlion_resumes/` to text.
3.  **Extraction:** The text is sent to `AIExtractor` (in `ai_extractor.py`), which uses prompts from `optimized_prompts.py` and validation from `AIValidator` (`ai_validator.py`).
4.  **Persistence:** Structured results are handed back to `main.py`, which calls `db.save_extraction()` in `db_manager.py`.
5.  **Intelligence:** `ml_engine.py` monitors the database. It scores the new extraction and flags it for review in the `review_dashboard.py` if the confidence is low.
6.  **Refinement:** When a user corrects data in the dashboard, `utils.py` (Feedback System) records the change, which `ml_engine.py` uses to "learn" for the next run.
