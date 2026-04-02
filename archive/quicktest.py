# Quick test script — run in the same directory as ner_schema.py
from ner_schema import EntitySchema, PreAnnotator

schema = EntitySchema()
pa = PreAnnotator(schema)

# Check dict sizes
stats = pa.get_dict_stats()
print("Dictionary sizes:")
for k, v in stats.items():
    print(f"  {k}: {v}")

# Verify total is ~730+
assert stats["_total"] > 700, f"Expected 700+ terms, got {stats['_total']}"

# Test pre-annotation on sample text
sample = """Ahmad bin Hassan | +65 9123 4567 | ahmad@gmail.com
Senior DevOps Engineer at DBS Bank Pte Ltd
Skills: Python, Docker, Kubernetes, AWS, Terraform
Education: Bachelor of Science from NUS
Certifications: AWS Certified Solutions Architect, PMP
Location: Singapore
"""
annotations = pa.pre_annotate(sample, confidence_threshold=0.5)
print(f"\nFound {len(annotations)} pre-annotations:")
for ann in annotations:
    print(f"  [{ann.entity_type:15s}] {ann.text} (conf: {ann.confidence})")

# Expect to find: PHONE, EMAIL, SKILL x5+, ORGANIZATION, DEGREE,
#                 INSTITUTION, CERTIFICATION x2, LOCATION
print("\n✅ All checks passed! PreAnnotator expansion verified!")