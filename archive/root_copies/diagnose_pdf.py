import pdfplumber
import PyPDF2

def diagnose_missing_text(pdf_path):
    """
    Comprehensive diagnosis of missing text
    """
    
    print("🔍 DIAGNOSING MISSING TEXT ISSUES")
    print("="*70)
    
    # === TEST 1: Basic Extraction ===
    print("\n📋 TEST 1: Basic Text Extraction")
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        text = page.extract_text()
        
        print(f"Characters extracted: {len(text)}")
        print(f"Preview (first 500 chars):")
        print(text[:500])
        print("...")
    
    # === TEST 2: Check for Text in Images ===
    print("\n🖼️ TEST 2: Checking Images for Text")
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        images = page.images
        
        print(f"Number of images: {len(images)}")
        
        if len(images) > 0:
            print("\n⚠️ IMAGES DETECTED - They might contain text!")
            print("Image details:")
            
            page_width = page.width
            page_height = page.height
            
            for i, img in enumerate(images[:5]):  # Show first 5
                img_width = img['width']
                img_height = img['height']
                coverage = (img_width / page_width) * (img_height / page_height) * 100
                
                print(f"  Image {i+1}:")
                print(f"    Size: {img_width:.0f}x{img_height:.0f} points")
                print(f"    Position: ({img['x0']:.0f}, {img['top']:.0f})")
                print(f"    Page coverage: {coverage:.1f}%")
                
                if coverage > 30:
                    print(f"    ⚠️ LARGE IMAGE - Likely contains important text!")
    
    # === TEST 3: Check Character Properties ===
    print("\n🔤 TEST 3: Checking Character Properties")
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        chars = page.chars
        
        if len(chars) > 0:
            # Check for invisible text (white on white)
            invisible_chars = []
            for char in chars:
                color = char.get('non_stroking_color', (0, 0, 0))
                # Check if color is close to white (1, 1, 1)
                if all(c > 0.9 for c in color):
                    invisible_chars.append(char)
            
            if invisible_chars:
                print(f"  ⚠️ Found {len(invisible_chars)} potentially invisible characters!")
                print(f"     (White/very light colored text)")
                sample_text = ''.join([c['text'] for c in invisible_chars[:50]])
                print(f"     Sample: {sample_text}")
            
            # Check for text outside page
            outside_chars = []
            for char in chars:
                if (char['x0'] < 0 or char['x0'] > page.width or
                    char['top'] < 0 or char['top'] > page.height):
                    outside_chars.append(char)
            
            if outside_chars:
                print(f"  ⚠️ Found {len(outside_chars)} characters outside page boundaries!")
    
    # === TEST 4: Check Form Fields ===
    print("\n📝 TEST 4: Checking Form Fields")
    with open(pdf_path, "rb") as file:
        pdf = PyPDF2.PdfReader(file)
        
        if pdf.is_encrypted:
            print("  🔒 PDF is encrypted/protected!")
        
        # Check for form fields
        if '/AcroForm' in pdf.trailer['/Root']:
            print("  📋 PDF contains form fields!")
            print("     Form field text might not be extracted by standard methods")
        else:
            print("  ✅ No form fields detected")
    
    # === TEST 5: Compare Multiple Extraction Methods ===
    print("\n⚖️ TEST 5: Comparing Extraction Methods")
    
    # Method 1: pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        text1 = pdf.pages[0].extract_text() or ""
    
    # Method 2: PyPDF2
    with open(pdf_path, "rb") as file:
        pdf = PyPDF2.PdfReader(file)
        text2 = pdf.pages[0].extract_text() or ""
    
    print(f"  pdfplumber extracted: {len(text1)} chars")
    print(f"  PyPDF2 extracted: {len(text2)} chars")
    print(f"  Difference: {abs(len(text1) - len(text2))} chars")
    
    if abs(len(text1) - len(text2)) > 100:
        print(f"  ⚠️ LARGE DIFFERENCE - Try different extraction methods!")
    
    # === RECOMMENDATIONS ===
    print("\n💡 RECOMMENDATIONS:")
    
    has_large_images = any(
        (img['width'] / page.width) * (img['height'] / page.height) > 0.3 
        for img in images
    )
    
    if has_large_images:
        print("  🔧 SOLUTION 1: Use OCR on images")
        print("     Some text is likely in images, not extractable directly")
        print("     Enable OCR extraction in your code")
    
    if len(chars) > 0 and len(text) < len(chars) / 2:
        print("  🔧 SOLUTION 2: Try different extraction settings")
        print("     Use layout-preserving extraction")
        print("     Try: page.extract_text(layout=True)")
    
    if len(images) > 50:
        print("  🔧 SOLUTION 3: This PDF has many embedded graphics")
        print("     Consider using Marker instead of pdfplumber")
        print("     Marker handles complex layouts better")
    
    print("\n" + "="*70)

# Run the diagnostic
diagnose_missing_text("merlion_resumes/50647_Mohamed Eilyar Bin Abdullah/Mohamed Eilyzar.pdf")