import pdfplumber
import PyPDF2
from pathlib import Path

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
inspect_pdf("merlion_resumes/48010_Jason JASON/Jason Teo Kok Heng (MCF).pdf")