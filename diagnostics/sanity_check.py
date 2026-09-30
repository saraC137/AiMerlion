
from transformers import pipeline

nlp = pipeline("ner", model="yashpwr/resume-ner-bert-v2", aggregation_strategy="simple")
text = "John Doe is a Software Engineer at Google. He graduated from Stanford University in 2020."
results = nlp(text)
for ent in results:
    print(f"{ent['entity_group']}: {ent['word']}")
