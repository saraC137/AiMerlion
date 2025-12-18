"""
🧪 OCR TEST SCRIPT
Save as: test_ocr.py
Usage: python test_ocr.py "path/to/problem.pdf"
"""

import sys
import os

def test_ocr_on_pdf(pdf_path):
    print("=" * 60)
    print("🔍 OCR TEST FOR OUTLINED/VECTOR PDFs")
    print("=" * 60)
    print(f"📄 Testing: {pdf_path}")
    print()
    
    # Step 1: Check pdf2image
    print("1️⃣ Checking pdf2image...")
    try:
        import pdf2image
        print("   ✅ pdf2image is installed!")
    except ImportError:
        print("   ❌ pdf2image NOT installed!")
        print("   Run: pip install pdf2image")
        return
    
    # Step 2: Check Poppler
    print("\n2️⃣ Checking Poppler...")
    import shutil
    if shutil.which("pdftoppm"):
        print("   ✅ Poppler is installed and in PATH!")
    else:
        print("   ❌ Poppler NOT found!")
        print("   Did you add it to PATH and restart terminal?")
        return
    
    # Step 3: Check pytesseract
    print("\n3️⃣ Checking Tesseract OCR...")
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"   ✅ Tesseract is installed! (v{version})")
    except Exception as e:
        print(f"   ❌ Tesseract error: {e}")
        return
    
    # Step 4: Actually try OCR!
    print("\n4️⃣ Converting PDF to images...")
    try:
        images = pdf2image.convert_from_path(
            pdf_path,
            first_page=1,
            last_page=1,  # Just first page for testing
            dpi=200
        )
        print(f"   ✅ Converted! Got {len(images)} image(s)")
    except Exception as e:
        print(f"   ❌ Conversion failed: {e}")
        return
    
    # Step 5: Run OCR
    print("\n5️⃣ Running OCR on first page...")
    try:
        from PIL import Image
        text = pytesseract.image_to_string(images[0], lang='eng')
        
        if text.strip():
            print("   ✅ OCR SUCCESSFUL!")
            print()
            print("   " + "=" * 50)
            print("   📝 EXTRACTED TEXT (first 500 chars):")
            print("   " + "=" * 50)
            # Show extracted text nicely formatted
            preview = text.strip()[:500]
            for line in preview.split('\n'):
                if line.strip():
                    print(f"   {line}")
            print("   " + "=" * 50)
        else:
            print("   ⚠️ OCR ran but extracted no text!")
            print("   The PDF might be very low quality or corrupted.")
            
    except Exception as e:
        print(f"   ❌ OCR failed: {e}")
        return
    
    print()
    print("🎉 TEST COMPLETE!")
    print()
    print("If you saw extracted text above, OCR is working!")
    print("Your main script should now handle these PDFs automatically.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_ocr.py <path_to_pdf>")
        print('Example: python test_ocr.py "C:\\resumes\\problem_resume.pdf"')
    else:
        test_ocr_on_pdf(sys.argv[1])