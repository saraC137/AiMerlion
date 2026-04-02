# Save as test_model.py
from ai_extractor_finetuned import AIExtractorFineTuned

extractor = AIExtractorFineTuned()

test_text = """
JOHN SMITH
Email: john.smith@example.com
Phone: +81-90-1234-5678
Date of Birth: 1990-05-15
"""

result = extractor.extract_all_fields(test_text)
print("Extracted:", result)