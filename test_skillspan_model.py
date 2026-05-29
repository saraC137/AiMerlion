# test_skillspan_model.py
import spacy

# Load your freshly trained model!
nlp = spacy.load("models/skillspan_knowledge")

# Test on resume-like text
sample_text = """
Senior Software Engineer with 5 years experience in Python, Docker, 
Kubernetes, and AWS. Proficient in React, Node.js, and PostgreSQL.
"""

doc = nlp(sample_text)
print("Knowledge entities found:")
for ent in doc.ents:
    print(f"  • {ent.text} ({ent.label_})")