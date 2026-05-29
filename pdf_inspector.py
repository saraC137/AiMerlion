import pdfplumber
import PyPDF2
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# ── Heuristic thresholds ──────────────────────────────────────────────────────
# A page must have at least this many characters to look like real text content.
# Anything lower than this on a per-page basis means there is likely image-based
# text that pdfplumber couldn't read.
MIN_CHARS_PER_PAGE = 150

# Total text below this is treated as effectively empty regardless of page count.
MIN_TOTAL_TEXT_CHARS = 100

# Suspicious-low text: total chars/page is between this and MIN_CHARS_PER_PAGE,
# so a thin text layer probably hides image content underneath.
SUSPICIOUS_CHARS_PER_PAGE = 400


def _count_images_deep(pdf_path: str) -> int:
    """
    Recursively count image XObjects across all pages, including images nested
    inside Form XObjects (which pdfplumber.page.images misses).

    Returns the total image XObject count, or 0 if scanning fails.
    """
    seen: set = set()
    image_count = 0

    def walk_xobject(xobj_dict):
        nonlocal image_count
        if not xobj_dict:
            return
        try:
            if hasattr(xobj_dict, 'get_object'):
                xobj_dict = xobj_dict.get_object()
            if not isinstance(xobj_dict, dict):
                return
            for name, obj in xobj_dict.items():
                if hasattr(obj, 'get_object'):
                    resolved = obj.get_object()
                else:
                    resolved = obj
                # Avoid cycles via indirect-object identity
                obj_id = id(resolved)
                if obj_id in seen:
                    continue
                seen.add(obj_id)

                subtype = resolved.get('/Subtype') if isinstance(resolved, dict) else None
                if subtype == '/Image':
                    image_count += 1
                elif subtype == '/Form':
                    # Form XObjects can themselves carry an /XObject resource
                    nested_resources = resolved.get('/Resources') if isinstance(resolved, dict) else None
                    if nested_resources:
                        if hasattr(nested_resources, 'get_object'):
                            nested_resources = nested_resources.get_object()
                        nested_xobj = nested_resources.get('/XObject') if isinstance(nested_resources, dict) else None
                        walk_xobject(nested_xobj)
        except Exception as e:
            logger.debug(f"XObject walk error: {e}")

    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    resources = page.get('/Resources')
                    if hasattr(resources, 'get_object'):
                        resources = resources.get_object()
                    if not isinstance(resources, dict):
                        continue
                    xobject = resources.get('/XObject')
                    walk_xobject(xobject)
                except Exception as e:
                    logger.debug(f"Page XObject scan error: {e}")
    except Exception as e:
        logger.debug(f"Deep image scan failed for {pdf_path}: {e}")
        return 0

    return image_count


def analyze_pdf_type(pdf_path: str) -> Dict[str, Any]:
    """
    Analyze PDF to determine its type and whether it contains images.

    Detection uses three independent signals — direct page.images, deep
    recursive XObject scan (catches images inside Form XObjects), and
    per-page text density — so mixed PDFs with hidden images and thin
    text layers are correctly flagged for OCR.

    Returns:
        dict with keys:
            - has_images: bool - True if PDF contains any images (any source)
            - has_text: bool - True if PDF has dense extractable text
            - has_fonts: bool - True if PDF has embedded fonts
            - has_vectors: bool - True if PDF has vector graphics
            - image_count: int - Max images found by either scanner
            - image_count_deep: int - Images found by recursive XObject scan
            - text_length: int - Length of extractable text
            - chars_per_page: float - Average extracted chars per page
            - page_count: int - Number of pages
            - pdf_type: str - 'text_only' | 'scanned' | 'mixed' | 'vector' | 'unclear'
            - needs_ocr: bool - True if OCR should be used
            - detection_reason: str - Human-readable explanation
    """
    result = {
        'has_images':        False,
        'has_text':          False,
        'has_fonts':         False,
        'has_vectors':       False,
        'image_count':       0,
        'image_count_deep':  0,
        'text_length':       0,
        'chars_per_page':    0.0,
        'page_count':        0,
        'pdf_type':          'unclear',
        'needs_ocr':         False,
        'detection_reason':  '',
    }

    try:
        # ── PASS 1: pdfplumber — text, top-level images, vectors ──────────────
        with pdfplumber.open(pdf_path) as pdf:
            total_images_shallow = 0
            total_text   = ""
            total_lines  = 0
            total_curves = 0
            page_count   = len(pdf.pages)

            for page in pdf.pages:
                total_images_shallow += len(page.images)
                page_text = page.extract_text() or ""
                total_text += page_text
                total_lines  += len(page.lines)
                total_curves += len(page.curves)

            result['page_count']     = max(page_count, 1)
            result['text_length']    = len(total_text)
            result['chars_per_page'] = len(total_text) / result['page_count']
            result['has_vectors']    = total_lines > 10 or total_curves > 5

        # ── PASS 2: PyPDF2 deep scan — catches Form-XObject-nested images ─────
        deep_image_count = _count_images_deep(pdf_path)
        result['image_count_deep'] = deep_image_count

        # Take the higher of the two image counts (most-thorough wins)
        result['image_count'] = max(total_images_shallow, deep_image_count)
        result['has_images']  = result['image_count'] > 0

        # ── PASS 3: Font detection (PyPDF2, page 1) ───────────────────────────
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                if len(pdf_reader.pages) > 0:
                    page = pdf_reader.pages[0]
                    resources = page.get('/Resources')
                    if resources:
                        if hasattr(resources, 'get_object'):
                            resources = resources.get_object()
                        result['has_fonts'] = (
                            '/Font' in resources if isinstance(resources, dict) else False
                        )
        except Exception as font_error:
            logger.debug(f"Font check failed: {font_error}")
            result['has_fonts'] = False

        # ── Text-density classification ───────────────────────────────────────
        # `has_text` now requires BOTH a minimum total AND a minimum density.
        # A 5-page PDF with 80 chars total → has_text=False (was True before).
        result['has_text'] = (
            result['text_length']    >= MIN_TOTAL_TEXT_CHARS
            and result['chars_per_page'] >= MIN_CHARS_PER_PAGE
        )

        # A thin text layer below SUSPICIOUS_CHARS_PER_PAGE → text exists but
        # is likely incomplete. We still flag for OCR even if no images detected.
        suspiciously_thin = (
            result['text_length']    >= MIN_TOTAL_TEXT_CHARS
            and result['chars_per_page'] < SUSPICIOUS_CHARS_PER_PAGE
        )

        # ── Decision matrix ───────────────────────────────────────────────────
        if result['has_text'] and result['has_fonts'] and not result['has_images'] and not suspiciously_thin:
            result['pdf_type']         = 'text_only'
            result['needs_ocr']        = False
            result['detection_reason'] = (
                f"text-only: {result['chars_per_page']:.0f} chars/page, no images, fonts present"
            )

        elif result['has_images'] and not result['has_text']:
            result['pdf_type']         = 'scanned'
            result['needs_ocr']        = True
            result['detection_reason'] = (
                f"scanned: {result['image_count']} images, only {result['text_length']} chars total"
            )

        elif result['has_images'] and result['has_text']:
            result['pdf_type']         = 'mixed'
            result['needs_ocr']        = True
            result['detection_reason'] = (
                f"mixed: {result['image_count']} images + {result['chars_per_page']:.0f} chars/page"
            )

        elif suspiciously_thin:
            # NEW BRANCH: text exists but density suggests image-embedded content
            result['pdf_type']         = 'mixed'
            result['needs_ocr']        = True
            result['detection_reason'] = (
                f"suspicious-thin text layer ({result['chars_per_page']:.0f} chars/page across "
                f"{result['page_count']} pages, threshold {SUSPICIOUS_CHARS_PER_PAGE}) — OCR likely needed"
            )

        elif result['has_vectors'] and not result['has_text']:
            result['pdf_type']         = 'vector'
            result['needs_ocr']        = True
            result['detection_reason'] = (
                f"vector: {total_lines} lines + {total_curves} curves, no real text"
            )

        else:
            result['pdf_type']         = 'unclear'
            result['needs_ocr']        = True
            result['detection_reason'] = "unclear signals — defaulting to OCR for safety"

        logger.debug(
            f"PDF Analysis for {Path(pdf_path).name}: type={result['pdf_type']}, "
            f"images={result['image_count']} (deep={result['image_count_deep']}, "
            f"shallow={total_images_shallow}), text={result['text_length']} chars "
            f"({result['chars_per_page']:.0f}/page), needs_ocr={result['needs_ocr']} — "
            f"{result['detection_reason']}"
        )

    except Exception as e:
        logger.warning(f"PDF analysis failed for {pdf_path}: {e}")
        # On error, default to using OCR to be safe
        result['needs_ocr']        = True
        result['pdf_type']         = 'error'
        result['detection_reason'] = f"inspection error: {e}"

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