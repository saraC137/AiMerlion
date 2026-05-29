
import logging
import coloredlogs
import json
from extraction.ai_extractor import AIExtractor
from extraction.ner_engine import get_ner_engine

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger,
                    fmt='%(asctime)s - 💅 %(levelname)s - %(message)s')

SAMPLE_RESUME = """
AHMAD BIN ABDULLAH
Senior Project Manager

--- EMPLOYMENT ---
1. BuildCorp SG | Project Manager | 2015 - Present
   * Led the construction of 'The Merlion Tower'.
2. KL Builders | Site Engineer | 2009 - 2015
   * Supervised 50+ workers daily.

--- EDUCATION ---
Master of Civil Engineering, Universiti Teknologi Malaysia (UTM), 2009
Bachelor of Civil Engineering, UTM, 2007
"""

def test_full_pipeline():
    """🧪 Test: Sectioning (AI) -> Entity Extraction (BERT NER)"""
    
    print("\n" + "="*80)
    print("🚀 TESTING FULL PIPELINE: AI SECTIONING + BERT NER")
    print("="*80)
    
    # 1. Initialize Engines
    ai_extractor = AIExtractor("llama3.1:8b", logger)
    ner_engine = get_ner_engine(logger)
    
    # 2. Section the Resume using the NEW AI method
    print("\n🔍 Stage 1: Segmenting resume with AI...")
    sections = ai_extractor.detect_sections_ai(SAMPLE_RESUME)
    
    # 3. Use BERT NER on key sections
    for section_name, content in sections.items():
        if section_name in ['experience', 'education']:
            print(f"\n📍 Processing Section: {section_name.upper()}")
            print("-" * 40)
            
            # Map section to BERT labels
            labels = []
            if section_name == 'experience':
                labels = ['Companies worked at', 'Designation', 'Years of Experience']
            elif section_name == 'education':
                labels = ['Degree', 'College Name', 'Graduation Year']
            
            # Extract via BERT
            raw_entities = ner_engine.extract_entities(content)
            print(f"🧩 RAW BERT Entities:")
            for ent in raw_entities:
                print(f"   • {ent['entity_group']}: {ent['word']} ({ent['score']:.2f})")
            
            entities = ner_engine.get_entities_by_type(content, labels)
            
            print(f"✅ BERT found these entities:")
            print(json.dumps(entities, indent=4))

    print("\n" + "="*80)
    print("✨ TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_full_pipeline()
