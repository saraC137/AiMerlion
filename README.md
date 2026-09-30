# AiMerlion: AI-Powered Resume Extraction System

## 1. Project Overview

AiMerlion is a sophisticated, standalone resume processing system designed to extract structured data from resumes with high accuracy. It leverages a powerful hybrid strategy that combines AI-first and Regex-first approaches to ensure both flexibility and reliability in data extraction.

The system is built with a resilient pre-processing pipeline that dynamically selects the best text extraction method (including multiple OCR fallbacks) and features robust error handling for AI interactions. AiMerlion is designed to be run from the command line and provides a user-friendly interactive interface.

## 2. Key Features

*   **Hybrid Extraction Strategy**: Utilizes an AI-first approach for contact information and a Regex-first approach for complex sections like work experience and education.
*   **Resilient Text Extraction**: Dynamically chooses the best text extraction method, with fallbacks to OCR for image-based or complex PDFs.
*   **AI-Powered Data Extraction**: Leverages Large Language Models (LLMs) to extract information from resumes.
*   **Interactive CLI**: Provides a user-friendly command-line interface for easy operation.
*   **Configurable**: Key parameters like file paths, AI models, and feature flags can be easily configured.
*   **Robust Error Handling**: Includes comprehensive error handling for AI interactions and file processing.
*   **Extensible**: The modular design allows for easy extension and customization.

## 3. Architecture

The system is orchestrated by the `UltimateResumeExtractor` class in `main.py`, which ties together the following key modules:

1.  **Configuration (`config.py`)**: A centralized file for all user-configurable parameters, including paths, AI model names, and feature flags.
2.  **Document Parser (`document_parser.py`)**: An intelligent module that handles text extraction from PDFs. It first attempts standard text extraction and falls back to OCR if the PDF contains non-selectable text, ensuring high-quality text for the extraction logic.
3.  **AI Extractor (`ai_extractor.py`)**: This is the core of the AI interaction. It implements the hybrid extraction strategy, using an AI-first approach for header/contact details and a Regex-first approach for structured data like work experience and skills. It contains the specific prompts and JSON schemas used to interact with the LLM.
4.  **Main (`main.py`)**: The entry point of the application. It provides an interactive command-line menu, orchestrates the entire extraction workflow, and handles report generation.

## 4. Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd AiMerlion
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The project has numerous dependencies. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## 5. Configuration

The primary configuration file is `config.py`. Before running the application, you can modify this file to suit your needs.

Key configuration options include:

*   `RESUME_FOLDER`: The path to the folder containing the resumes to be processed.
*   `FAST_MODEL` and `SMART_MODEL`: The names of the Ollama models to be used for AI extraction.
*   `USE_MARKER_PDF`: A feature flag to enable or disable an advanced PDF parsing strategy.
*   `USE_AI_EXTRACTION`: A feature flag to enable or disable AI-powered extraction.

## 6. Usage

Run everything from the repo root as a module (`python -m package.script`), so
package imports and relative data paths (`resume_extractions.db`, `models/`) resolve.

```bash
python -m core.main              # resume extraction pipeline (interactive menu)
python -m annotation             # NER annotation tool -> http://localhost:5055
python -m analysis.evaluate_lite # baseline vs fine-tuned evaluation
```

## 7. Project Structure

Code is grouped by task:

```
AiMerlion/
├── annotation/        # NER annotation tool (Flask, port 5055)
│   ├── app.py         #   app, config, NER components, DB connection
│   ├── queries.py     #   read-only DB helpers
│   ├── routes/        #   annotate, annotators, export, classify, iaa
│   └── templates/     #   index.html, annotate.html + index/ and annotate/ partials
├── core/              # pipeline entry point, config, utils
├── extraction/        # regex+AI extractor, document/PDF parsing, NER schema, SG normalizer
├── data/              # DB manager, synthetic data generation, golden test set
├── data_prep/         # raw text extraction, dataset importers, training data prep
├── ml/                # fine-tuning, NER/classifier training, GGUF export, role predictor, vector search
├── analysis/          # evaluation, run comparison, quality audits, exports
├── dashboards/        # review + classification dashboards, accuracy validator API
├── diagnostics/       # health checks, DB maintenance, one-off probes
├── tests/             # test scripts (pytest: tests/test_sg_normalizer.py)
├── archive/           # old versions; root_copies/ and replaced/ hold pre-reorg duplicates
└── annotation_tool.py # launcher kept for `python annotation_tool.py`
```

## 8. Key Modules

*   `core/main.py`: pipeline entry point.
*   `extraction/ai_extractor.py`: hybrid regex + AI field extraction.
*   `extraction/document_parser.py`: document parsing with OCR fallback.
*   `extraction/ner_schema.py`: entity schema, tokenizer, BIO tagger, annotation storage, IAA.
*   `annotation/`: web tool for labelling resume entities.
*   `analysis/evaluate_lite.py`: scores extractors against the golden test set.
