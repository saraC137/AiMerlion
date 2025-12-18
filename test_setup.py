"""
✨ PDF Setup Verification Script for WINDOWS ✨
Run this to check if everything is installed correctly!
"""

import shutil
import platform

print("=" * 60)
print("🔍 PDF EXTRACTION SETUP CHECK - WINDOWS EDITION")
print("=" * 60)
print(f"📍 Operating System: {platform.system()} {platform.release()}")
print()

all_good = True

# Check 1: pdfplumber
print("1️⃣ Checking pdfplumber...")
try:
    import pdfplumber
    print("   ✅ pdfplumber - INSTALLED")
except ImportError:
    print("   ❌ pdfplumber - MISSING!")
    print("      Run: pip install pdfplumber")
    all_good = False

# Check 2: Pillow
print("\n2️⃣ Checking Pillow (PIL)...")
try:
    from PIL import Image
    print("   ✅ Pillow - INSTALLED")
except ImportError:
    print("   ❌ Pillow - MISSING!")
    print("      Run: pip install Pillow")
    all_good = False

# Check 3: pytesseract
print("\n3️⃣ Checking pytesseract...")
try:
    import pytesseract
    print("   ✅ pytesseract (Python package) - INSTALLED")
    
    # Try to actually use Tesseract
    try:
        version = pytesseract.get_tesseract_version()
        print(f"   ✅ Tesseract OCR engine - INSTALLED (v{version})")
    except Exception as e:
        print("   ❌ Tesseract OCR engine - NOT FOUND!")
        print("      Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("      Make sure to add it to PATH during installation!")
        all_good = False
        
except ImportError:
    print("   ❌ pytesseract - MISSING!")
    print("      Run: pip install pytesseract")
    all_good = False

# Check 4: pdf2image
print("\n4️⃣ Checking pdf2image...")
try:
    import pdf2image
    print("   ✅ pdf2image - INSTALLED")
except ImportError:
    print("   ❌ pdf2image - MISSING!")
    print("      Run: pip install pdf2image")
    all_good = False

# Check 5: Poppler (the CRITICAL one for Windows!)
print("\n5️⃣ Checking Poppler (required by pdf2image)...")
poppler_path = shutil.which("pdftoppm")
if poppler_path:
    print(f"   ✅ Poppler - INSTALLED at {poppler_path}")
else:
    print("   ❌ Poppler - NOT FOUND IN PATH!")
    print("      ")
    print("      📥 TO INSTALL POPPLER ON WINDOWS:")
    print("      1. Download from: https://github.com/oschwann/poppler-windows/releases")
    print("      2. Extract to: C:\\Program Files\\poppler\\")
    print("      3. Add to PATH: C:\\Program Files\\poppler\\Library\\bin")
    print("      4. Restart your terminal!")
    all_good = False

# Final verdict
print()
print("=" * 60)
if all_good:
    print("🎉 ALL CHECKS PASSED! You're ready to extract PDFs, queen! 👑")
else:
    print("⚠️  SOME CHECKS FAILED! Please install missing components above.")
print("=" * 60)
