import pdfplumber
import PyPDF2
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def analyze_pdf_type(pdf_path: str) -> Dict[str, Any]:
    """
    Analyze PDF to determine its type and whether it contains images.

    Returns:
        dict with keys:
            - has_images: bool - True if PDF contains any images
            - has_text: bool - True if PDF has extractable text (>50 chars)
            - has_fonts: bool - True if PDF has embedded fonts
            - has_vectors: bool - True if PDF has vector graphics
            - image_count: int - Number of images found
            - text_length: int - Length of extractable text
            - pdf_type: str - 'text_only', 'scanned', 'mixed', 'vector', 'unclear'
            - needs_ocr: bool - True if OCR should be used
    """
    result = {
        'has_images': False,
        'has_text': False,
        'has_fonts': False,
        'has_vectors': False,
        'image_count': 0,
        'text_length': 0,
        'pdf_type': 'unclear',
        'needs_ocr': False
    }

    try:
        # Analyze with pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            total_images = 0
            total_text = ""
            total_lines = 0
            total_curves = 0

            for page in pdf.pages:
                # Count images across all pages
                images = page.images
                total_images += len(images)

                # Get text
                page_text = page.extract_text() or ""
                total_text += page_text

                # Count vectors
                total_lines += len(page.lines)
                total_curves += len(page.curves)

            result['image_count'] = total_images
            result['has_images'] = total_images > 0
            result['text_length'] = len(total_text)
            result['has_text'] = len(total_text) > 50
            result['has_vectors'] = total_lines > 10 or total_curves > 5

        # Check fonts with PyPDF2
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                if len(pdf_reader.pages) > 0:
                    page = pdf_reader.pages[0]
                    resources = page.get('/Resources')
                    if resources:
                        # Handle IndirectObject by resolving it
                        if hasattr(resources, 'get_object'):
                            resources = resources.get_object()
                        result['has_fonts'] = '/Font' in resources if isinstance(resources, dict) else False
                    else:
                        result['has_fonts'] = False
        except Exception as font_error:
            logger.debug(f"Font check failed: {font_error}")
            result['has_fonts'] = False

        # Determine PDF type and OCR need
        if result['has_text'] and result['has_fonts'] and not result['has_images']:
            result['pdf_type'] = 'text_only'
            result['needs_ocr'] = False  # Pure text PDF - no OCR needed
        elif result['has_images'] and not result['has_text']:
            result['pdf_type'] = 'scanned'
            result['needs_ocr'] = True  # Scanned PDF - OCR required
        elif result['has_images'] and result['has_text']:
            result['pdf_type'] = 'mixed'
            result['needs_ocr'] = True  # Mixed PDF - use OCR to capture image text
        elif result['has_vectors'] and not result['has_text']:
            result['pdf_type'] = 'vector'
            result['needs_ocr'] = True  # Vector/shape-based - try OCR
        else:
            result['pdf_type'] = 'unclear'
            result['needs_ocr'] = True  # When in doubt, use OCR

        logger.debug(f"PDF Analysis for {Path(pdf_path).name}: type={result['pdf_type']}, "
                    f"images={result['image_count']}, text={result['text_length']} chars, needs_ocr={result['needs_ocr']}")

    except Exception as e:
        logger.warning(f"PDF analysis failed for {pdf_path}: {e}")
        # On error, default to using OCR to be safe
        result['needs_ocr'] = True
        result['pdf_type'] = 'error'

    return result


def inspect_pdf(pdf_path: str):
    """
    Interactive PDF inspection tool
    """
    
    print(f"\n🔍 INSPECTING: {Path(pdf_path).name}")
    print("="*70)
    
    # === BASIC INFO ===
    with open(pdf_path, "rb") as file:
        pdf = PyPDF2.PdfReader(file)
        print(f"\n📄 BASIC INFO:")
        print(f"   Total Pages: {len(pdf.pages)}")
        
        # Check if encrypted
        if pdf.is_encrypted:
            print(f"   🔒 Encrypted: YES")
        else:
            print(f"   🔓 Encrypted: NO")
        
        # Metadata
        info = pdf.metadata
        if info:
            print(f"   Creator: {info.get('/Creator', 'Unknown')}")
            print(f"   Producer: {info.get('/Producer', 'Unknown')}")
    
    # === FIRST PAGE ANALYSIS ===
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        
        print(f"\n📃 FIRST PAGE ANALYSIS:")
        print(f"   Dimensions: {page.width:.0f} x {page.height:.0f} points")
        
        # Text
        text = page.extract_text() or ""
        print(f"\n   📝 TEXT:")
        print(f"      Extractable: {'YES ✅' if len(text) > 50 else 'NO ❌'}")
        print(f"      Length: {len(text)} characters")
        if text:
            print(f"      Preview: {text[:100]}...")
        
        # Characters
        chars = page.chars
        print(f"\n   🔤 CHARACTERS:")
        print(f"      Count: {len(chars)}")
        if chars:
            print(f"      Sample: {chars[0] if chars else 'None'}")
        
        # Images
        images = page.images
        print(f"\n   🖼️ IMAGES:")
        print(f"      Count: {len(images)}")
        if images:
            for i, img in enumerate(images[:3]):  # Show first 3
                print(f"      Image {i+1}: {img['width']}x{img['height']} at ({img['x0']:.0f}, {img['top']:.0f})")
        
        # Lines (vector graphics)
        lines = page.lines
        print(f"\n   📏 LINES/VECTORS:")
        print(f"      Count: {len(lines)}")
        
        # Rectangles
        rects = page.rects
        print(f"      Rectangles: {len(rects)}")
        
        # Curves
        curves = page.curves
        print(f"      Curves: {len(curves)}")
    
    # === FONT ANALYSIS ===
    with open(pdf_path, "rb") as file:
        pdf = PyPDF2.PdfReader(file)
        page = pdf.pages[0]
        
        print(f"\n   🔠 FONTS:")
        if '/Font' in page.get('/Resources', {}):
            fonts = page['/Resources']['/Font']
            print(f"      Count: {len(fonts)}")
            for font_name in list(fonts.keys())[:5]:  # Show first 5
                print(f"      - {font_name}")
        else:
            print(f"      Count: 0 (No fonts found)")
    
    # === DETERMINATION ===
    print(f"\n🎯 DETERMINATION:")
    
    has_text = len(text) > 50
    has_chars = len(chars) > 20
    has_fonts = '/Font' in page.get('/Resources', {})
    has_images = len(images) > 0
    has_vectors = len(lines) > 10 or len(curves) > 5
    
    if has_text and has_fonts and not has_images:
        print(f"   ✅ TRUE TEXT PDF")
        print(f"   📌 Extraction Method: Marker or PDFPlumber")
    
    elif has_images and not has_text:
        print(f"   📸 SCANNED IMAGE PDF")
        print(f"   📌 Extraction Method: OCR Required!")
    
    elif has_images and has_text:
        print(f"   📄 MIXED PDF (Text + Images)")
        print(f"   📌 Extraction Method: Marker (best), or PDFPlumber")
    
    elif has_vectors and not has_text:
        print(f"   ⚠️ VECTOR/SHAPE-BASED PDF")
        print(f"   📌 Extraction Method: Try Marker, then OCR")
    
    else:
        print(f"   ❓ UNCLEAR TYPE")
        print(f"   📌 Extraction Method: Try all methods")
    
    print("="*70 + "\n")

# === USE IT ===
if __name__ == "__main__":
    inspect_pdf("merlion_resumes/48010_Jason JASON/Jason Teo Kok Heng (MCF).pdf")