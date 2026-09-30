
from transformers import pipeline

nlp = pipeline("ner", model="yashpwr/resume-ner-bert-v2")
print("Labels:", nlp.model.config.id2label)
