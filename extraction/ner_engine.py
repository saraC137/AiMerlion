
import logging
from typing import List, Dict, Any, Optional
from transformers import pipeline
import torch

class NEREngine:
    """
    🚀 BERT-based NER Engine for Resume Entity Extraction
    Uses: yashpwr/resume-ner-bert-v2
    """
    
    _instance = None
    _model_name = "yashpwr/resume-ner-bert-v2"

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.device = 0 if torch.cuda.is_available() else -1
        self.nlp = None
        self.logger.info(f"🧬 NER Engine initialized (Device: {'GPU' if self.device == 0 else 'CPU'})")

    def _load_model(self):
        """Lazy load the transformer pipeline"""
        if self.nlp is None:
            self.logger.info(f"⏳ Loading BERT NER model: {self._model_name}...")
            try:
                self.nlp = pipeline(
                    "ner", 
                    model=self._model_name, 
                    aggregation_strategy="simple",
                    device=self.device
                )
                self.logger.info("✅ BERT NER model loaded successfully!")
            except Exception as e:
                self.logger.error(f"❌ Failed to load BERT model: {e}")
                raise

    def _clean_text_for_ner(self, text: str) -> str:
        """
        🧹 Clean structural noise for BERT while preserving semantic context.
        Replaces pipes with commas and removes dividers.
        """
        import re
        # Remove divider lines
        text = re.sub(r'[-=]{3,}', '', text)
        
        # Replace pipes and strange bullets with commas/spaces
        text = text.replace('|', ',')
        text = re.sub(r'^\s*[\*\+•-]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+[\.\)]\s*', '', text, flags=re.MULTILINE)
        
        # Normalize whitespace but keep newlines
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities line-by-line and apply post-processing fixes.
        Line-by-line processing helps BERT focus on individual resume entries.
        """
        if not text or len(text.strip()) < 5:
            return []

        self._load_model()
        
        # 🧹 Pre-clean but keep lines
        cleaned_text = self._clean_text_for_ner(text)
        lines = cleaned_text.split('\n')
        
        all_results = []
        for line in lines:
            if not line.strip(): continue
            try:
                line_results = self.nlp(line.strip())
                all_results.extend(line_results)
            except Exception as e:
                self.logger.error(f"💥 NER Inference failed for line: {e}")

        # 🛠️ Post-process results
        fixed_results = []
        for ent in all_results:
            word = ent['word'].strip().strip('"').strip("'").strip()
            group = ent['entity_group']
            
            # Fix common misclassifications
            if group == 'Degree' and any(role in word.lower() for role in ['manager', 'engineer', 'developer', 'analyst']):
                group = 'Designation'
            
            if group == 'Companies worked at' and any(sch in word.lower() for sch in ['university', 'college', 'polytechnic']):
                group = 'College Name'
            
            # Filter out garbage words
            if word.lower() in ['from', 'in', 'at', 'the', 'and', 'with', 'for', 'to', 'of']:
                continue
            
            if len(word) < 2:
                continue
                
            ent['word'] = word
            ent['entity_group'] = group
            fixed_results.append(ent)
            
        return fixed_results

    def get_entities_by_type(self, text: str, entity_types: List[str]) -> Dict[str, List[str]]:
        """
        Extract specific types of entities (e.g. ['Companies worked at', 'Designation'])
        Returns a map of entity_type -> list of found strings.
        """
        raw_results = self.extract_entities(text)
        organized = {etype: [] for etype in entity_types}
        
        for entity in raw_results:
            group = entity.get('entity_group')
            if group in entity_types:
                word = entity.get('word', '').strip()
                if word and word not in organized[group]:
                    organized[group].append(word)
                    
        return organized

# Global instance getter
def get_ner_engine(logger=None):
    global _ner_instance
    if '_ner_instance' not in globals() or globals()['_ner_instance'] is None:
        globals()['_ner_instance'] = NEREngine(logger)
    return globals()['_ner_instance']
