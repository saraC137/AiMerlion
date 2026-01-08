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

The main entry point for the application is `main.py`. You can run it from the command line:

```bash
python main.py
```

This will launch an interactive menu with the following options:

*   Process resumes
*   Generate reports
*   Diagnose AI
*   Exit

## 7. Project Structure

```
AiMerlion/
├───merlion_resumes/      # Input resumes folder
├───main.py               # Main application entry point
├───ai_extractor.py       # Core AI extraction logic
├───document_parser.py    # PDF text extraction and OCR
├───config.py             # Project configuration
├───requirements.txt      # Project dependencies
├───utils.py              # Utility functions
├───evaluator.py          # (Not fully analyzed) Tools for evaluating extraction quality
├───finetune_model.py     # (Not fully analyzed) Scripts for fine-tuning AI models
├───labeling_tool.py      # (Not fully analyzed) A tool for labeling data
└───...
```

## 8. Key Modules

*   `main.py`: The main entry point of the application.
*   `ai_extractor.py`: Handles the AI-powered data extraction from resumes.
*   `document_parser.py`: Responsible for parsing various document formats, including PDFs with OCR fallback.
*   `config.py`: Centralized configuration for the project.
*   `evaluator.py`: Provides tools to evaluate the performance of the extraction process.
*   `finetune_model.py`: Contains scripts for fine-tuning the AI models used in the project.
*   `labeling_tool.py`: A tool to assist in labeling resume data for model training.
*   `utils.py`: A collection of utility functions used across the project.
```