"""
Test Marker Integration
"""

import os
import sys
import logging
import coloredlogs
from marker_extractor import MarkerExtractor

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger,
                   fmt='%(asctime)s - [TEST] %(levelname)s - %(message)s')

def test_marker():
    """Test Marker with a sample resume"""
    print("\n" + "="*80)
    print("MARKER PDF EXTRACTION TEST".center(80))
    print("="*80 + "\n")

    # Initialize Marker
    print("📦 Initializing Marker extractor...")
    extractor = MarkerExtractor()

    if not extractor.available:
        print("❌ Marker not available. Cannot proceed with test.")
        return

    print("✅ Marker initialized successfully!\n")

    # Test with a sample PDF
    test_files = [
        "merlion_resumes/44233_Unknown_Candidate/Reyner_Tan_SystemsLead.pdf",
        "merlion_resumes/45885_ANG KAI WIN/Kai Win Ang Cloud Engineer (Level 2)- Singaporean Only Resume.pdf",
    ]

    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"⚠️ File not found: {test_file}")
            continue

        print(f"\n{'='*80}")
        print(f"📄 Testing: {os.path.basename(test_file)}")
        print(f"{'='*80}\n")

        # Extract text
        text = extractor.extract_pdf(test_file)

        if text:
            print("✅ Extraction successful!")

            # Get stats
            stats = extractor.get_extraction_stats(text)
            print("\n📊 Extraction Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")

            # Show preview
            print("\n📄 Text Preview (first 800 characters):")
            print("-" * 80)
            print(text[:800])
            print("-" * 80)

            # Check for key resume sections
            print("\n🔍 Section Detection:")
            sections = {
                "Email": "@" in text,
                "Phone": any(p in text.lower() for p in ["phone", "tel", "mobile"]),
                "Experience": any(e in text.lower() for e in ["experience", "work history", "employment"]),
                "Education": any(e in text.lower() for e in ["education", "university", "degree"]),
                "Skills": "skill" in text.lower(),
            }

            for section, found in sections.items():
                emoji = "✅" if found else "❌"
                print(f"   {emoji} {section}: {'Found' if found else 'Not found'}")

            break  # Test with first available file

        else:
            print("❌ Extraction failed!")

    print("\n" + "="*80)
    print("🏁 TEST COMPLETE".center(80))
    print("="*80 + "\n")

if __name__ == "__main__":
    test_marker()
