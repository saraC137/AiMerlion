
import logging
import coloredlogs
from extraction.ner_engine import get_ner_engine

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger,
                    fmt='%(asctime)s - 💅 %(levelname)s - %(message)s')

def test_ner_only():
    print("🧬 Initializing NER Engine...")
    ner = get_ner_engine(logger)
    
    text = "Software Engineer at Google from 2021 to 2024. Previously worked at Amazon."
    print(f"🔍 Extracting from: {text}")
    
    entities = ner.extract_entities(text)
    for ent in entities:
        print(f"📍 {ent['entity_group']}: {ent['word']} (Score: {ent['score']:.2f})")

if __name__ == "__main__":
    test_ner_only()
