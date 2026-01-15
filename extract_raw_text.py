"""
Raw Text Extraction from PDFs and DOCX files using Marker + OCR Hybrid
Extracts and saves raw text from all resume files for inspection
Uses PDF inspector to determine extraction strategy:
- Text-only PDFs: Marker only (fast)
- PDFs with images: Hybrid (Marker + OCR merged)
- DOCX files: python-docx extraction
"""

import os
import sys
import logging
from pathlib import Path
import coloredlogs
import pandas as pd
from datetime import datetime
from typing import Optional
from difflib import SequenceMatcher
import numpy as np
from marker_extractor import get_marker_extractor
from pdf_inspector import analyze_pdf_type

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger,
                   fmt='%(asctime)s - [RAW] %(levelname)s - %(message)s')


class HybridExtractor:
    """
    Hybrid PDF extractor that combines Marker + OCR for complete text extraction
    """

    def __init__(self):
        self.marker_extractor = get_marker_extractor()
        self.ocr_available = False
        self.ocr_engine = None
        self.ocr_instance = None
        self._initialize_ocr()

    def _initialize_ocr(self):
        """Initialize OCR engine"""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.ocr_available = True
            self.ocr_engine = 'pytesseract'
            self.ocr_instance = pytesseract
            logger.info("OCR engine ready (pytesseract)")
        except Exception as e:
            logger.warning(f"Pytesseract not available: {e}")

            # Try PaddleOCR as fallback
            try:
                from paddleocr import PaddleOCR
                self.ocr_instance = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                self.ocr_available = True
                self.ocr_engine = 'paddleocr'
                logger.info("OCR engine ready (PaddleOCR)")
            except Exception as e2:
                logger.warning(f"PaddleOCR not available: {e2}")
                logger.warning("No OCR engine available - hybrid extraction will be limited")

    def extract_with_ocr(self, file_path: str, dpi: int = 300) -> Optional[str]:
        """Extract text from PDF using OCR"""
        if not self.ocr_available:
            return None

        file_name = os.path.basename(file_path)

        try:
            import pdf2image
            logger.info(f"Running OCR on {file_name} (DPI: {dpi})...")

            images = pdf2image.convert_from_path(file_path, dpi=dpi)
            logger.info(f"   Converted {len(images)} pages to images")

            all_text = []

            for i, img in enumerate(images):
                page_text = ""

                if self.ocr_engine == 'pytesseract':
                    page_text = self.ocr_instance.image_to_string(img, lang='eng')

                elif self.ocr_engine == 'paddleocr':
                    img_array = np.array(img)
                    result = self.ocr_instance.ocr(img_array, cls=True)
                    if result and result[0]:
                        for line in result[0]:
                            if line and len(line) >= 2:
                                text = line[1][0] if isinstance(line[1], tuple) else line[1]
                                page_text += str(text) + "\n"

                if page_text.strip():
                    all_text.append(page_text.strip())
                    logger.info(f"   Page {i+1}: {len(page_text.split())} words")

            combined_text = "\n\n".join(all_text)

            if combined_text.strip():
                logger.info(f"OCR complete: {len(combined_text)} chars, {len(combined_text.split())} words")
                return combined_text
            else:
                logger.warning(f"OCR produced no text")
                return None

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return None

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts (0.0 to 1.0)"""
        if not text1 or not text2:
            return 0.0
        if text1 == text2:
            return 1.0
        if text1 in text2 or text2 in text1:
            return 0.9
        return SequenceMatcher(None, text1, text2).ratio()

    def merge_texts(self, marker_text: str, ocr_text: str, file_name: str) -> str:
        """
        Concatenate Marker extraction + OCR results
        Keeps ALL text from both sources to ensure nothing is missed
        """
        if not marker_text and not ocr_text:
            return ""
        if not marker_text:
            return ocr_text
        if not ocr_text:
            return marker_text

        logger.info(f"Concatenating: Marker ({len(marker_text)} chars) + OCR ({len(ocr_text)} chars)")

        # Concatenate both - don't filter anything out
        merged = marker_text.strip() + "\n\n" + "="*80 + "\n"
        merged += "=== FULL OCR TEXT (may contain duplicates) ===\n"
        merged += "="*80 + "\n\n"
        merged += ocr_text.strip()

        logger.info(f"Concatenated result: {len(merged)} total chars")
        return merged

    def extract_hybrid(self, file_path: str) -> Optional[str]:
        """
        HYBRID EXTRACTION: Combines Marker + OCR for complete text capture

        1. Extract with Marker (gets selectable text)
        2. Extract with OCR (gets image-based text)
        3. Merge both results intelligently
        """
        file_name = os.path.basename(file_path)
        logger.info(f"HYBRID EXTRACTION for {file_name}")

        # Step 1: Get text from Marker (selectable text)
        logger.info("Step 1: Extracting with Marker...")
        marker_text = None
        if self.marker_extractor and self.marker_extractor.available:
            marker_text = self.marker_extractor.extract_pdf(file_path, use_ocr_fallback=False)
            if marker_text:
                logger.info(f"   Marker: {len(marker_text)} chars extracted")
            else:
                logger.warning(f"   Marker: No text extracted")

        # Step 2: Get text from OCR (image-based text)
        logger.info("Step 2: Extracting with OCR...")
        ocr_text = self.extract_with_ocr(file_path)
        if ocr_text:
            logger.info(f"   OCR: {len(ocr_text)} chars extracted")
        else:
            logger.warning(f"   OCR: No text extracted")

        # Step 3: Merge results
        logger.info("Step 3: Merging results...")
        merged_text = self.merge_texts(marker_text, ocr_text, file_name)

        if merged_text:
            logger.info(f"HYBRID RESULT: {len(merged_text)} total chars")
        else:
            logger.warning(f"HYBRID EXTRACTION produced no text")

        return merged_text

    def extract_text_only(self, file_path: str) -> Optional[str]:
        """
        TEXT-ONLY EXTRACTION: Uses Marker only (fast, for text-only PDFs)
        """
        file_name = os.path.basename(file_path)
        logger.info(f"TEXT-ONLY EXTRACTION for {file_name}")

        if self.marker_extractor and self.marker_extractor.available:
            text = self.marker_extractor.extract_pdf(file_path, use_ocr_fallback=False)
            if text:
                logger.info(f"Marker extracted {len(text)} chars")
            return text

        return None

    def extract_docx(self, file_path: str) -> Optional[str]:
        """
        DOCX EXTRACTION: Extract text from Word documents using python-docx
        Handles standard paragraphs/tables AND text boxes/shapes (common in resumes)
        """
        file_name = os.path.basename(file_path)
        logger.info(f"DOCX EXTRACTION for {file_name}")

        try:
            from docx import Document
        except ImportError:
            logger.error("python-docx not installed. Install with: pip install python-docx")
            return None

        try:
            doc = Document(file_path)
            all_text = []

            # Extract text from paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    all_text.append(para.text)

            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        all_text.append(" | ".join(row_text))

            combined_text = "\n".join(all_text)

            # If standard extraction found nothing, try XML-based extraction
            # This handles text boxes, shapes, and other non-standard layouts
            if not combined_text.strip():
                logger.info("Standard extraction found no text, trying XML-based extraction...")
                combined_text = self._extract_docx_from_xml(file_path)

            if combined_text and combined_text.strip():
                logger.info(f"DOCX extracted: {len(combined_text)} chars, {len(combined_text.split())} words")
                return combined_text
            else:
                logger.warning(f"DOCX extraction produced no text")
                return None

        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return None

    def _extract_docx_from_xml(self, file_path: str) -> Optional[str]:
        """
        Extract text directly from DOCX XML when standard extraction fails.
        Handles text boxes, shapes, drawings, and other non-standard layouts.
        """
        import zipfile
        import re

        try:
            all_texts = []

            with zipfile.ZipFile(file_path, 'r') as z:
                # Process main document
                if 'word/document.xml' in z.namelist():
                    content = z.read('word/document.xml').decode('utf-8')
                    texts = self._extract_text_from_xml_content(content)
                    all_texts.extend(texts)

                # Also check headers/footers for additional content
                for name in z.namelist():
                    if name.startswith('word/header') or name.startswith('word/footer'):
                        content = z.read(name).decode('utf-8')
                        texts = self._extract_text_from_xml_content(content)
                        all_texts.extend(texts)

            if all_texts:
                # Join text elements, handling spacing intelligently
                combined = self._join_xml_text_elements(all_texts)
                logger.info(f"XML extraction found {len(combined)} chars from {len(all_texts)} text elements")
                return combined

            return None

        except Exception as e:
            logger.error(f"XML-based DOCX extraction failed: {e}")
            return None

    def _extract_text_from_xml_content(self, xml_content: str) -> list:
        """Extract all <w:t> text elements from XML content"""
        import re
        # Match <w:t> elements, including those with attributes like xml:space="preserve"
        texts = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', xml_content)
        return texts

    def _join_xml_text_elements(self, texts: list) -> str:
        """
        Intelligently join XML text elements to reconstruct readable text.
        Handles spacing and line breaks based on content patterns.
        """
        import re

        if not texts:
            return ""

        result = []
        current_line = []

        for text in texts:
            # Skip empty or whitespace-only elements
            if not text or text.isspace():
                # If we have content in current line, this might be a separator
                if current_line and text and len(text) > 3:
                    # Multiple spaces often indicate new section
                    result.append(' '.join(current_line))
                    current_line = []
                continue

            current_line.append(text)

            # Check for natural line endings
            if text.endswith((':',)) or any(text.strip().endswith(x) for x in ['.', '!', '?']):
                # Might be end of a section
                pass

        # Add any remaining content
        if current_line:
            result.append(' '.join(current_line))

        # Join lines with newlines and clean up excessive whitespace
        combined = '\n'.join(result)
        # Clean up multiple consecutive spaces
        combined = re.sub(r' +', ' ', combined)
        # Clean up multiple consecutive newlines
        combined = re.sub(r'\n\s*\n', '\n\n', combined)

        return combined.strip()

    def get_extraction_stats(self, text: str) -> dict:
        """Get statistics about extracted text"""
        if self.marker_extractor:
            return self.marker_extractor.get_extraction_stats(text)
        return {
            "total_chars": len(text) if text else 0,
            "total_lines": len(text.split('\n')) if text else 0,
            "total_words": len(text.split()) if text else 0,
            "sections_found": 0,
            "tables_detected": False,
            "table_rows": 0,
            "lists_detected": False,
            "list_items": 0,
        }


def get_extraction_limit():
    """
    Interactive menu to get how many candidates to extract

    Returns:
        int: Number of candidates to extract (0 means all)
    """
    print("\n" + "="*80)
    print("RAW TEXT EXTRACTION - OPTIONS MENU".center(80))
    print("="*80 + "\n")

    print("How many candidates do you want to extract?")
    print("\n  1. Extract ALL candidates")
    print("  2. Extract first 10 candidates")
    print("  3. Extract first 25 candidates")
    print("  4. Extract first 50 candidates")
    print("  5. Extract first 100 candidates")
    print("  6. Enter a custom number")

    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()

            if choice == '1':
                return 0  # 0 means extract all
            elif choice == '2':
                return 10
            elif choice == '3':
                return 25
            elif choice == '4':
                return 50
            elif choice == '5':
                return 100
            elif choice == '6':
                custom = input("Enter number of candidates to extract: ").strip()
                num = int(custom)
                if num > 0:
                    return num
                else:
                    print("Please enter a positive number!")
            else:
                print("Invalid choice! Please enter a number between 1 and 6.")
        except ValueError:
            print("Invalid input! Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            sys.exit(0)


def extract_raw_text(resume_folder: str = "merlion_resumes", output_folder: str = "raw_text_output", limit: int = 0):
    """
    Extract raw text from all PDFs and DOCX files and save to text files and Excel

    Args:
        resume_folder: Folder containing resume PDFs and DOCX files
        output_folder: Folder to save raw text files
        limit: Maximum number of candidates to extract (0 = extract all)
    """
    print("\n" + "="*80)
    print("RAW TEXT EXTRACTION (PDF + DOCX)".center(80))
    print("="*80 + "\n")

    # Show extraction limit
    if limit > 0:
        logger.info(f"Extraction limit: {limit} candidates")
    else:
        logger.info(f"Extraction limit: ALL candidates")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"Output folder: {output_folder}")

    # Initialize Hybrid Extractor
    logger.info("Initializing Hybrid Extractor (Marker + OCR)...")
    extractor = HybridExtractor()

    if not extractor.marker_extractor or not extractor.marker_extractor.available:
        logger.error("Marker not available. Cannot proceed.")
        return

    logger.info("Hybrid Extractor initialized!\n")

    # Find all PDFs and DOCX files
    resume_files = []
    for root, dirs, files in os.walk(resume_folder):
        for file in files:
            if file.lower().endswith('.pdf') or file.lower().endswith('.docx'):
                resume_files.append(os.path.join(root, file))

    total_found = len(resume_files)

    # Apply limit if specified
    if limit > 0 and limit < len(resume_files):
        resume_files = resume_files[:limit]
        logger.info(f"Found {total_found} files (PDF/DOCX), processing first {limit}\n")
    else:
        logger.info(f"Found {len(resume_files)} files (PDF/DOCX) to process\n")

    if not resume_files:
        logger.warning("No PDF or DOCX files found!")
        return

    # Process each file (PDF or DOCX)
    successful = 0
    failed = 0
    extraction_data = []  # Store data for Excel generation

    for idx, file_path in enumerate(resume_files, 1):
        file_name = os.path.basename(file_path)
        folder_name = os.path.basename(os.path.dirname(file_path))
        is_docx = file_name.lower().endswith('.docx')

        print(f"\n{'='*80}")
        print(f"[{idx}/{len(resume_files)}] Processing: {folder_name}/{file_name}")
        print(f"{'='*80}")

        try:
            # Handle DOCX files
            if is_docx:
                logger.info(f"DOCX file detected - using python-docx extraction")
                text = extractor.extract_docx(file_path)
                pdf_info = {
                    'pdf_type': 'DOCX',
                    'image_count': 0,
                    'text_length': len(text) if text else 0,
                    'needs_ocr': False
                }
            else:
                # Step 1: Inspect PDF to determine extraction strategy
                pdf_info = analyze_pdf_type(file_path)
                logger.info(f"PDF Inspector: type={pdf_info['pdf_type']}, "
                           f"images={pdf_info['image_count']}, text={pdf_info['text_length']} chars, "
                           f"needs_ocr={pdf_info['needs_ocr']}")

                # Step 2: Extract based on PDF type
                if pdf_info['needs_ocr']:
                    # PDF has images - use HYBRID extraction (Marker + OCR merged)
                    logger.info(f"Images detected ({pdf_info['image_count']}) - using HYBRID extraction")
                    text = extractor.extract_hybrid(file_path)
                else:
                    # Text-only PDF - use Marker only (faster)
                    logger.info(f"Text-only PDF - using Marker extraction")
                    text = extractor.extract_text_only(file_path)

            if text and len(text.strip()) >= 50:
                # Create output filename
                output_name = f"{folder_name}_{os.path.splitext(file_name)[0]}.txt"
                output_path = os.path.join(output_folder, output_name)

                # Determine extraction method description
                if is_docx:
                    extraction_method = 'python-docx'
                elif pdf_info['needs_ocr']:
                    extraction_method = 'HYBRID (Marker + OCR)'
                else:
                    extraction_method = 'Marker only'

                # Save raw text
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"Source: {file_path}\n")
                    f.write(f"File Type: {pdf_info['pdf_type']}\n")
                    f.write(f"Images Found: {pdf_info['image_count']}\n")
                    f.write(f"Extraction Method: {extraction_method}\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(text)

                # Get stats
                stats = extractor.get_extraction_stats(text)

                logger.info(f"Extracted successfully!")
                logger.info(f"   Characters: {stats['total_chars']:,}")
                logger.info(f"   Words: {stats['total_words']:,}")
                logger.info(f"   Lines: {stats['total_lines']:,}")
                logger.info(f"   Sections: {stats['sections_found']}")
                logger.info(f"   Tables: {stats['table_rows']} rows" if stats['tables_detected'] else "   Tables: None")
                logger.info(f"   Lists: {stats['list_items']} items" if stats['lists_detected'] else "   Lists: None")
                logger.info(f"   Saved to: {output_name}")

                # Show preview
                preview = text[:300].replace('\n', ' ')
                logger.info(f"   Preview: {preview}...")

                # Add to extraction data for Excel
                extraction_data.append({
                    'Candidate_ID': idx,
                    'Folder_Name': folder_name,
                    'File_Name': file_name,
                    'File_Path': file_path,
                    'File_Type': pdf_info['pdf_type'],
                    'Images_Found': pdf_info['image_count'],
                    'Extraction_Method': 'DOCX' if is_docx else ('HYBRID' if pdf_info['needs_ocr'] else 'Marker'),
                    'Characters': stats['total_chars'],
                    'Words': stats['total_words'],
                    'Lines': stats['total_lines'],
                    'Sections': stats['sections_found'],
                    'Table_Rows': stats['table_rows'] if stats['tables_detected'] else 0,
                    'List_Items': stats['list_items'] if stats['lists_detected'] else 0,
                    'Raw_Text': text
                })

                successful += 1
            else:
                logger.warning(f"Extraction produced too little text (< 50 chars)")
                failed += 1

        except Exception as e:
            logger.error(f"Failed to extract {file_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            failed += 1

    # Summary
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE".center(80))
    print("="*80)
    print(f"\nSuccessful: {successful}")
    print(f"Failed: {failed}")
    print(f"Raw text files saved to: {output_folder}\n")

    # Generate Excel file if we have data
    if extraction_data:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_filename = f"raw_text_extraction_{timestamp}.xlsx"
            excel_path = os.path.join(output_folder, excel_filename)

            logger.info(f"Generating Excel file...")

            # Create DataFrame
            df = pd.DataFrame(extraction_data)

            # Create Excel writer with formatting
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Write main data
                df.to_excel(writer, sheet_name='Extracted Text', index=False)

                # Get workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets['Extracted Text']

                # Auto-adjust column widths (except Raw_Text)
                for col_idx, col in enumerate(df.columns, 1):
                    if col != 'Raw_Text':
                        max_length = max(
                            df[col].astype(str).apply(len).max(),
                            len(col)
                        )
                        col_letter = chr(64 + col_idx) if col_idx <= 26 else 'A' + chr(64 + col_idx - 26)
                        worksheet.column_dimensions[col_letter].width = min(max_length + 2, 50)
                    else:
                        # Set Raw_Text column to a reasonable width
                        col_letter = chr(64 + col_idx) if col_idx <= 26 else 'A' + chr(64 + col_idx - 26)
                        worksheet.column_dimensions[col_letter].width = 100

            logger.info(f"Excel file generated successfully!")
            logger.info(f"   File: {excel_filename}")
            logger.info(f"   Contains {len(extraction_data)} extracted resumes")
            print(f"\nExcel file saved: {excel_path}\n")

        except Exception as e:
            logger.error(f"Failed to generate Excel file: {e}")
    else:
        logger.warning("No data to export to Excel")


if __name__ == "__main__":
    import config

    # Get extraction limit from user
    limit = get_extraction_limit()

    # Run extraction
    extract_raw_text(
        resume_folder=config.RESUME_FOLDER,
        output_folder="raw_text_output",
        limit=limit
    )
