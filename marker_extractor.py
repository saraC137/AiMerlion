"""
🌟 Marker-based PDF Extraction for Resume Processing 🌟
Uses Marker to convert PDFs to clean markdown with preserved structure
Includes OCR fallback for scanned/image-based PDFs
"""

import os
import logging
from typing import Optional, Dict
from pathlib import Path
import coloredlogs
import numpy as np

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger,
                   fmt='%(asctime)s - 📄 %(levelname)s - %(message)s')

# Try to import config for OCR settings
try:
    import config
except ImportError:
    config = None


class MarkerExtractor:
    """
    📄 Marker-powered PDF extractor!
    Converts PDFs to high-quality markdown for better AI extraction
    Includes OCR fallback for scanned/image-based PDFs
    """

    def __init__(self):
        """
        Initialize Marker PDF extractor with OCR fallback
        """
        self.available = False
        self.converter = None
        self.ocr_available = False
        self.ocr_engine = None
        self.ocr_instance = None

        # Initialize Marker
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict

            logger.info("🔄 Loading Marker models (this may take a moment on first run)...")

            # Create the model dictionary (artifact_dict)
            models = create_model_dict()

            logger.info("🔄 Initializing PDF converter...")

            # Create the PDF converter with the models
            self.converter = PdfConverter(
                artifact_dict=models,
                processor_list=None,  # Use default processors
                renderer=None,  # Use default renderer
                config=None  # Use default config
            )

            self.available = True
            logger.info("✅ Marker initialized successfully!")
            logger.info("   📚 Models loaded and ready for PDF extraction")

        except ImportError as e:
            logger.error(f"❌ marker-pdf package not installed: {e}")
            logger.error("   Install with: pip install marker-pdf")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Marker: {e}")
            logger.error("   Falling back to pdfplumber extraction")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")

        # Initialize OCR as fallback
        self._initialize_ocr()

    def _initialize_ocr(self):
        """Initialize OCR engine for fallback extraction"""
        ocr_engine = getattr(config, 'OCR_ENGINE', 'pytesseract') if config else 'pytesseract'

        # Try pytesseract first
        if ocr_engine == 'pytesseract':
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                self.ocr_available = True
                self.ocr_engine = 'pytesseract'
                self.ocr_instance = pytesseract
                logger.info("✅ OCR fallback ready (pytesseract)")
                return
            except Exception as e:
                logger.debug(f"Pytesseract not available: {e}")

        # Try PaddleOCR
        if ocr_engine == 'paddleocr' or not self.ocr_available:
            try:
                from paddleocr import PaddleOCR
                lang = getattr(config, 'OCR_LANGUAGE', 'en') if config else 'en'
                paddle_lang = 'en' if lang == 'eng' else lang
                self.ocr_instance = PaddleOCR(use_angle_cls=True, lang=paddle_lang, show_log=False)
                self.ocr_available = True
                self.ocr_engine = 'paddleocr'
                logger.info("✅ OCR fallback ready (PaddleOCR)")
                return
            except Exception as e:
                logger.debug(f"PaddleOCR not available: {e}")

        # Try EasyOCR
        if ocr_engine == 'easyocr' or not self.ocr_available:
            try:
                import easyocr
                lang = getattr(config, 'OCR_LANGUAGE', 'en') if config else 'en'
                easy_lang = ['en'] if lang == 'eng' else [lang]
                self.ocr_instance = easyocr.Reader(easy_lang, gpu=False)
                self.ocr_available = True
                self.ocr_engine = 'easyocr'
                logger.info("✅ OCR fallback ready (EasyOCR)")
                return
            except Exception as e:
                logger.debug(f"EasyOCR not available: {e}")

        if not self.ocr_available:
            logger.warning("⚠️ No OCR engine available for fallback")

    def extract_with_ocr(self, file_path: str) -> Optional[str]:
        """
        📸 Extract text from PDF using OCR
        Uses the same simple approach as test_ocr.py for reliability
        """
        if not self.ocr_available:
            return None

        file_name = os.path.basename(file_path)
        ocr_dpi = getattr(config, 'OCR_DPI', 200) if config else 200  # 200 DPI like test_ocr.py

        try:
            import pdf2image
            logger.info(f"📸 Running OCR on {file_name} using {self.ocr_engine} (DPI: {ocr_dpi})...")

            # Convert PDF to images (same as test_ocr.py)
            images = pdf2image.convert_from_path(file_path, dpi=ocr_dpi)
            logger.info(f"   🖼️ Converted {len(images)} pages to images")

            all_text = []

            for i, img in enumerate(images):
                logger.info(f"   📄 OCR processing page {i+1}/{len(images)}")
                page_text = ""

                if self.ocr_engine == 'pytesseract':
                    # Simple call like test_ocr.py - NO extra config for reliability!
                    page_text = self.ocr_instance.image_to_string(img, lang='eng')

                elif self.ocr_engine == 'paddleocr':
                    img_array = np.array(img)
                    result = self.ocr_instance.ocr(img_array, cls=True)
                    # Extract ALL text (no confidence filter)
                    if result and result[0]:
                        for line in result[0]:
                            if line and len(line) >= 2:
                                text = line[1][0] if isinstance(line[1], tuple) else line[1]
                                page_text += str(text) + "\n"

                elif self.ocr_engine == 'easyocr':
                    img_array = np.array(img)
                    result = self.ocr_instance.readtext(img_array)
                    # Extract ALL text (no confidence filter)
                    for detection in result:
                        text = detection[1]
                        page_text += text + "\n"
                else:
                    page_text = ""

                if page_text.strip():
                    all_text.append(page_text.strip())
                    word_count = len(page_text.split())
                    logger.info(f"      ✅ Page {i+1}: {word_count} words extracted")
                else:
                    logger.warning(f"      ⚠️ Page {i+1}: No text extracted")

            combined_text = "\n\n".join(all_text)

            if combined_text.strip():
                total_words = len(combined_text.split())
                logger.info(f"✅ OCR complete: {len(combined_text)} chars, {total_words} words from {file_name}")
                return combined_text
            else:
                logger.warning(f"⚠️ OCR produced no text from {file_name}")
                return None

        except Exception as e:
            logger.error(f"❌ OCR failed for {file_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def extract_pdf(self, file_path: str, use_ocr_fallback: bool = True) -> Optional[str]:
        """
        Extract text from PDF using Marker with OCR fallback

        Args:
            file_path: Path to the PDF file
            use_ocr_fallback: Whether to try OCR if Marker fails

        Returns:
            Extracted text as markdown, or None if extraction fails
        """
        if not os.path.exists(file_path):
            logger.error(f"❌ File not found: {file_path}")
            return None

        file_name = os.path.basename(file_path)
        ocr_min_threshold = getattr(config, 'OCR_MIN_TEXT_THRESHOLD', 100) if config else 100
        full_text = None

        # Try Marker first if available
        if self.available:
            try:
                logger.info(f"📄 Extracting {file_name} with Marker...")

                # Convert PDF to markdown using Marker's new API
                rendered = self.converter(file_path)

                # Extract the markdown text
                full_text = rendered.markdown if hasattr(rendered, 'markdown') else str(rendered)

                if full_text and len(full_text.strip()) >= ocr_min_threshold:
                    logger.info(f"✅ Marker extracted {len(full_text)} characters from {file_name}")

                    # Log extraction quality
                    if hasattr(rendered, 'metadata'):
                        metadata = rendered.metadata
                        if hasattr(metadata, 'pages'):
                            logger.info(f"   📊 Pages processed: {metadata.pages}")

                    return full_text
                else:
                    logger.warning(f"⚠️ Marker extracted insufficient text ({len(full_text.strip()) if full_text else 0} chars) from {file_name}")

            except Exception as e:
                logger.error(f"❌ Marker extraction failed for {file_name}: {e}")
                import traceback
                logger.debug(f"   Traceback: {traceback.format_exc()}")

        # Try OCR fallback if Marker failed or produced little text
        if use_ocr_fallback and self.ocr_available:
            if not full_text or len(full_text.strip()) < ocr_min_threshold:
                logger.info(f"🔄 Trying OCR fallback for {file_name}...")
                ocr_text = self.extract_with_ocr(file_path)

                if ocr_text:
                    # Use OCR text if it's better than what we have
                    if not full_text or len(ocr_text.strip()) > len(full_text.strip()):
                        logger.info(f"✅ OCR produced better result ({len(ocr_text)} chars vs {len(full_text.strip()) if full_text else 0})")
                        return ocr_text
                    else:
                        logger.info(f"📄 Keeping Marker result (better than OCR)")

        return full_text

    def get_extraction_stats(self, text: str) -> Dict:
        """
        Get statistics about the extracted markdown text

        Args:
            text: Extracted markdown text

        Returns:
            Dictionary with statistics
        """
        if not text:
            return {}

        lines = text.split('\n')
        words = text.split()

        # Detect sections (markdown headers)
        sections = [line for line in lines if line.startswith('#')]

        # Detect tables (markdown tables use |)
        table_lines = [line for line in lines if '|' in line and line.count('|') >= 2]

        # Detect lists
        list_lines = [line for line in lines if line.strip().startswith(('-', '*', '+'))]

        # Detect code blocks
        code_blocks = text.count('```')

        return {
            "total_chars": len(text),
            "total_lines": len(lines),
            "total_words": len(words),
            "sections_found": len(sections),
            "tables_detected": len(table_lines) > 0,
            "table_rows": len(table_lines),
            "lists_detected": len(list_lines) > 0,
            "list_items": len(list_lines),
            "code_blocks": code_blocks // 2 if code_blocks > 0 else 0,
        }


# Global instance (initialized once for performance)
_marker_instance = None

def get_marker_extractor() -> MarkerExtractor:
    """
    Get or create the global Marker extractor instance
    This ensures models are only loaded once
    """
    global _marker_instance
    if _marker_instance is None:
        _marker_instance = MarkerExtractor()
    return _marker_instance
