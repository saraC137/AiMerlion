"""
Raw Text Extraction from PDFs using Marker + OCR Hybrid
Extracts and saves raw text from all resume PDFs for inspection
Uses PDF inspector to determine extraction strategy:
- Text-only PDFs: Marker only (fast)
- PDFs with images: Hybrid (Marker + OCR merged)
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
    Extract raw text from all PDFs and save to text files and Excel

    Args:
        resume_folder: Folder containing resume PDFs
        output_folder: Folder to save raw text files
        limit: Maximum number of candidates to extract (0 = extract all)
    """
    print("\n" + "="*80)
    print("RAW TEXT EXTRACTION WITH HYBRID METHOD".center(80))
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

    # Find all PDFs
    pdf_files = []
    for root, dirs, files in os.walk(resume_folder):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))

    total_found = len(pdf_files)

    # Apply limit if specified
    if limit > 0 and limit < len(pdf_files):
        pdf_files = pdf_files[:limit]
        logger.info(f"Found {total_found} PDF files, processing first {limit}\n")
    else:
        logger.info(f"Found {len(pdf_files)} PDF files to process\n")

    if not pdf_files:
        logger.warning("No PDF files found!")
        return

    # Process each PDF
    successful = 0
    failed = 0
    extraction_data = []  # Store data for Excel generation

    for idx, pdf_path in enumerate(pdf_files, 1):
        file_name = os.path.basename(pdf_path)
        folder_name = os.path.basename(os.path.dirname(pdf_path))

        print(f"\n{'='*80}")
        print(f"[{idx}/{len(pdf_files)}] Processing: {folder_name}/{file_name}")
        print(f"{'='*80}")

        try:
            # Step 1: Inspect PDF to determine extraction strategy
            pdf_info = analyze_pdf_type(pdf_path)
            logger.info(f"PDF Inspector: type={pdf_info['pdf_type']}, "
                       f"images={pdf_info['image_count']}, text={pdf_info['text_length']} chars, "
                       f"needs_ocr={pdf_info['needs_ocr']}")

            # Step 2: Extract based on PDF type
            if pdf_info['needs_ocr']:
                # PDF has images - use HYBRID extraction (Marker + OCR merged)
                logger.info(f"Images detected ({pdf_info['image_count']}) - using HYBRID extraction")
                text = extractor.extract_hybrid(pdf_path)
            else:
                # Text-only PDF - use Marker only (faster)
                logger.info(f"Text-only PDF - using Marker extraction")
                text = extractor.extract_text_only(pdf_path)

            if text and len(text.strip()) >= 50:
                # Create output filename
                output_name = f"{folder_name}_{os.path.splitext(file_name)[0]}.txt"
                output_path = os.path.join(output_folder, output_name)

                # Save raw text
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"Source: {pdf_path}\n")
                    f.write(f"PDF Type: {pdf_info['pdf_type']}\n")
                    f.write(f"Images Found: {pdf_info['image_count']}\n")
                    f.write(f"Extraction Method: {'HYBRID (Marker + OCR)' if pdf_info['needs_ocr'] else 'Marker only'}\n")
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
                    'PDF_Path': pdf_path,
                    'PDF_Type': pdf_info['pdf_type'],
                    'Images_Found': pdf_info['image_count'],
                    'Extraction_Method': 'HYBRID' if pdf_info['needs_ocr'] else 'Marker',
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
