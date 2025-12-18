# test_pdf_setup.py - Run this to check your setup! 💅

print("🔍 Checking PDF extraction setup...\n")

# Check 1: pdfplumber
try:
    import pdfplumber
    print("✅ pdfplumber - INSTALLED")
except ImportError:
    print("❌ pdfplumber - MISSING! Run: pip install pdfplumber")

# Check 2: pytesseract
try:
    import pytesseract
    # Try to get tesseract version (this confirms it's actually working)
    version = pytesseract.get_tesseract_version()
    print(f"✅ pytesseract - INSTALLED (Tesseract v{version})")
except ImportError:
    print("❌ pytesseract - MISSING! Run: pip install pytesseract")
except Exception as e:
    print(f"⚠️ pytesseract installed but Tesseract OCR not found!")
    print(f"   Run: sudo apt-get install tesseract-ocr")

# Check 3: pdf2image
try:
    import pdf2image
    print("✅ pdf2image - INSTALLED")
except ImportError:
    print("❌ pdf2image - MISSING! Run: pip install pdf2image")

# Check 4: Poppler (needed by pdf2image)
import shutil
if shutil.which("pdftoppm"):
    print("✅ Poppler - INSTALLED")
else:
    print("❌ Poppler - MISSING! Run: sudo apt-get install poppler-utils")

# Check 5: PIL/Pillow
try:
    from PIL import Image
    print("✅ Pillow - INSTALLED")
except ImportError:
    print("❌ Pillow - MISSING! Run: pip install Pillow")

print("\n✨ Setup check complete!")