"""
Raw Text Extraction from PDFs using Marker
Extracts and saves raw text from all resume PDFs for inspection
"""

import os
import sys
import logging
from pathlib import Path
import coloredlogs
import pandas as pd
from datetime import datetime
from marker_extractor import get_marker_extractor

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger,
                   fmt='%(asctime)s - [RAW] %(levelname)s - %(message)s')


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
                    print("❌ Please enter a positive number!")
            else:
                print("❌ Invalid choice! Please enter a number between 1 and 6.")
        except ValueError:
            print("❌ Invalid input! Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n\n❌ Operation cancelled by user.")
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
    print("RAW TEXT EXTRACTION WITH MARKER".center(80))
    print("="*80 + "\n")

    # Show extraction limit
    if limit > 0:
        logger.info(f"📊 Extraction limit: {limit} candidates")
    else:
        logger.info(f"📊 Extraction limit: ALL candidates")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"📁 Output folder: {output_folder}")

    # Initialize Marker
    logger.info("📦 Initializing Marker extractor...")
    extractor = get_marker_extractor()

    if not extractor.available:
        logger.error("❌ Marker not available. Cannot proceed.")
        return

    logger.info("✅ Marker initialized successfully!\n")

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
        logger.info(f"📄 Found {total_found} PDF files, processing first {limit}\n")
    else:
        logger.info(f"📄 Found {len(pdf_files)} PDF files to process\n")

    if not pdf_files:
        logger.warning("⚠️ No PDF files found!")
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
            # Extract text
            text = extractor.extract_pdf(pdf_path)

            if text and len(text.strip()) >= 50:
                # Create output filename
                output_name = f"{folder_name}_{os.path.splitext(file_name)[0]}.txt"
                output_path = os.path.join(output_folder, output_name)

                # Save raw text
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"Source: {pdf_path}\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(text)

                # Get stats
                stats = extractor.get_extraction_stats(text)

                logger.info(f"✅ Extracted successfully!")
                logger.info(f"   📊 Characters: {stats['total_chars']:,}")
                logger.info(f"   📊 Words: {stats['total_words']:,}")
                logger.info(f"   📊 Lines: {stats['total_lines']:,}")
                logger.info(f"   📊 Sections: {stats['sections_found']}")
                logger.info(f"   📊 Tables: {stats['table_rows']} rows" if stats['tables_detected'] else "   📊 Tables: None")
                logger.info(f"   📊 Lists: {stats['list_items']} items" if stats['lists_detected'] else "   📊 Lists: None")
                logger.info(f"   💾 Saved to: {output_name}")

                # Show preview
                preview = text[:300].replace('\n', ' ')
                logger.info(f"   📄 Preview: {preview}...")

                # Add to extraction data for Excel
                extraction_data.append({
                    'Candidate_ID': idx,
                    'Folder_Name': folder_name,
                    'File_Name': file_name,
                    'PDF_Path': pdf_path,
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
                logger.warning(f"⚠️ Extraction produced too little text (< 50 chars)")
                failed += 1

        except Exception as e:
            logger.error(f"❌ Failed to extract {file_name}: {e}")
            failed += 1

    # Summary
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE".center(80))
    print("="*80)
    print(f"\n✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Raw text files saved to: {output_folder}\n")

    # Generate Excel file if we have data
    if extraction_data:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_filename = f"raw_text_extraction_{timestamp}.xlsx"
            excel_path = os.path.join(output_folder, excel_filename)

            logger.info(f"📊 Generating Excel file...")

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
                for idx, col in enumerate(df.columns, 1):
                    if col != 'Raw_Text':
                        max_length = max(
                            df[col].astype(str).apply(len).max(),
                            len(col)
                        )
                        worksheet.column_dimensions[chr(64 + idx)].width = min(max_length + 2, 50)
                    else:
                        # Set Raw_Text column to a reasonable width
                        worksheet.column_dimensions[chr(64 + idx)].width = 100

            logger.info(f"✅ Excel file generated successfully!")
            logger.info(f"   📄 File: {excel_filename}")
            logger.info(f"   📊 Contains {len(extraction_data)} extracted resumes")
            print(f"\n📊 Excel file saved: {excel_path}\n")

        except Exception as e:
            logger.error(f"❌ Failed to generate Excel file: {e}")
    else:
        logger.warning("⚠️ No data to export to Excel")


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
