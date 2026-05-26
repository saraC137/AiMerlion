# 🩺 Act 0 — Diagnostic Gate

## Selected cases

- **case_01**: ID `100194` | chinese | Lim Chin Peng
- **case_02**: ID `100118` | chinese | Li Jing Qiang (Mr)
- **case_03**: ID `44144` | chinese | Khoo Yan Jie
- **case_04**: ID `101576` | chinese | Li Wenbo
- **case_05**: ID `44101` | malay | Siti Suliana binte Zainudin
- **case_06**: ID `60628` | malay | Mhd Hafiz bin Ismail
- **case_07**: ID `99987` | malay | Anugerah Khalis binti Mohd Sabta
- **case_08**: ID `43940` | indian | Jairinder Singh Chahal
- **case_09**: ID `43724` | indian | Rajminder Singh (Mr)
- **case_10**: ID `100049` | other | Preethikrupa Joseph Ravi


## 📋 How to complete Act 0

For each case:

1. **Run your current pipeline** on the raw text:
   ```python
   # Pseudocode — adjust to your ai_extractor entry point
   from ai_extractor import extract_resume
   raw = open('diagnostic/case_01_raw.txt').read()
   # strip the header comments (lines starting with #)
   raw_clean = '\n'.join(l for l in raw.split('\n') if not l.startswith('#'))
   output = extract_resume(raw_clean)
   ```

2. **Paste the model output** into `case_XX_template.json` → `model_output` section

3. **Read the raw text yourself** and fill in `ground_truth` with what
   SHOULD have been extracted

4. **For each field, fill in `field_evaluation`**:
   - `correct`: true / false
   - `failure_type`: hallucination / truncation / missing / wrong_format / wrong_assignment
   - `root_cause`: ONE of:
     - `llm_weakness` — model didn't understand the pattern
     - `post_processing` — extracted right, code mangled it
     - `schema_issue` — JSON structure rejected valid output
     - `prompt_issue` — system prompt didn't ask for it correctly
     - `ocr_issue` — input text was already corrupted

5. **Decision gate** (after all 10 cases):
   - Count total failures by root_cause
   - If >60% are `llm_weakness` → ✅ Curriculum training is the right move
   - If >40% are `post_processing` or `prompt_issue` → 🛑 Fix code first, not model
   - If >30% are `schema_issue` → 🛑 Fix schema first
   - If >20% are `ocr_issue` → 🛑 Fix input pipeline first

## 🎯 Time estimate
- Running the model: 5 min total
- Filling templates: 20-30 min for all 10 cases (2-3 min each)
- Tallying decision: 5 min

**Total: ~30-40 minutes of focused work.**

💋 — Fairy Codemother
