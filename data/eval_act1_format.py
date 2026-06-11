"""
eval_act1_format.py — Act 1 format-compliance REPORT CARD.

Runs your fine-tuned model (via Ollama) over the FROZEN golden set and scores
exactly what Act 1 is graded on: JSON parse rate, schema adherence, hallucination,
field leakage. Logs every raw output so failures are attributable.

This uses the firewall test set (golden_sg_my.conformed.jsonl) — never trained on.

Usage:
    python eval_act1_format.py
Outputs:
    act1_scoreboard.csv     — per-record pass/fail (your field x act scoreboard starts here)
    act1_raw_outputs.jsonl  — raw model text per record (for attribution / debugging)
"""

import json, re, urllib.request
from collections import Counter

# ════════════════ EDIT HERE ════════════════
OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL       = "aimerlion_act1_v2"                 # your Ollama model name (see export step)
GOLDEN      = "golden.jsonl"
RESULTS_CSV = "act1_scoreboard.csv"
RAW_LOG     = "act1_raw_outputs.jsonl"
TEMPERATURE = 0.0                              # deterministic for extraction
NUM_CTX     = 16384                            # golden_021/022 are ~8-10k prompt tokens; 8192 errors out

# MUST be byte-for-byte the INSTRUCTION you trained with (prepare_lite_for_training.py)
INSTRUCTION = """You are a resume extraction model specialized in Singapore and Malaysia resumes. Extract the candidate's information from the resume text below into a JSON object using this exact schema.

Schema (use these keys exactly; empty string "" or empty list [] when a field is absent — never invent values):
- Name, Phone, Email, Current Location, Current Company, Current Title, Summary, Function (string)
- Industry, Language Skills, Certifications, hard_skills/tags, soft_skills/skills, Achievements (lists)
- Work Experience: list of {company, title, from, to, responsibility (list)}
- Project Experience: list
- Education: list of {school, major, degree, dates}

Pay special attention to:
- Patronymics: bin/binti (Malay), s/o or d/o (Indian)
- Singapore 8-digit phones (+65 8XXX XXXX / 9XXX XXXX); Malaysia phones (+60 1X-XXX XXXX)
- Company suffixes: Pte Ltd, Sdn Bhd, Berhad
- CMFAS module numbers (Module 1A, 5, 6, 6A, 8, 8A, 9, 9A, HI) -> Certifications
- Singapore/Malaysia polytechnics and universities

Output ONLY the JSON object, no commentary."""

EXTRACT_FIELDS = [
    "Name", "Phone", "Email", "Current Location", "Current Company", "Current Title",
    "Summary", "Function", "Industry", "Language Skills",
    "Certifications", "hard_skills/tags", "soft_skills/skills", "Achievements",
    "Work Experience", "Project Experience", "Education",
]
LIST_FIELDS = {"Industry", "Language Skills", "Certifications", "hard_skills/tags",
               "soft_skills/skills", "Achievements", "Work Experience",
               "Project Experience", "Education"}
SCALAR_FIELDS = [f for f in EXTRACT_FIELDS if f not in LIST_FIELDS]
HALLUC_EXCLUDE = {"Certifications"}   # golden has 0/25 certs annotated — gap, not ground truth
LEAK_MARKERS = ["see description", "see above", "see resume", "see cv",
                "n/a", "not available", "refer to"]


def call_model(raw_text: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": raw_text},
        ],
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
    }
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read()).get("message", {}).get("content", "")


def extract_json(text: str):
    text = re.sub(r"```(?:json)?", "", text or "")
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def is_empty(v):
    return v in (None, "", [], {})


def score_one(pred, gt):
    if pred is None:
        return {"parse_ok": False, "schema_ok": False, "type_ok": False,
                "missing": EXTRACT_FIELDS[:], "extra": [], "hallucinations": [], "leaks": []}
    keys = set(pred.keys())
    missing = sorted(set(EXTRACT_FIELDS) - keys)
    extra = sorted(keys - set(EXTRACT_FIELDS))
    type_ok = all(isinstance(pred.get(f), list) for f in LIST_FIELDS if f in pred)
    halluc = [f for f in EXTRACT_FIELDS
              if f not in HALLUC_EXCLUDE and is_empty(gt.get(f)) and not is_empty(pred.get(f))]
    leaks = [f for f in SCALAR_FIELDS
             if any(m in str(pred.get(f, "")).lower() for m in LEAK_MARKERS)]
    return {"parse_ok": True, "schema_ok": (not missing and not extra), "type_ok": type_ok,
            "missing": missing, "extra": extra, "hallucinations": halluc, "leaks": leaks}


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def main():
    rows = [json.loads(l) for l in open(GOLDEN, encoding="utf-8") if l.strip()]
    n = len(rows)
    parse = schema = type_ok = 0
    halluc_total = leak_total = 0
    contact_hits = contact_total = 0   # bonus content peek (Email + Phone)

    with open(RAW_LOG, "w", encoding="utf-8") as raw_f, \
         open(RESULTS_CSV, "w", encoding="utf-8") as csv_f:
        csv_f.write("id,parse_ok,schema_ok,type_ok,n_halluc,n_leaks,missing,extra\n")
        for r in rows:
            gt = r["ground_truth"]
            out = call_model(r["raw_text"])
            raw_f.write(json.dumps({"id": r["id"], "raw_output": out}, ensure_ascii=False) + "\n")
            pred = extract_json(out)
            s = score_one(pred, gt)
            parse += s["parse_ok"]; schema += s["schema_ok"]; type_ok += s["type_ok"]
            halluc_total += len(s["hallucinations"]); leak_total += len(s["leaks"])
            if pred:
                for f in ("Email", "Phone"):
                    if not is_empty(gt.get(f)):
                        contact_total += 1
                        contact_hits += (norm(pred.get(f, "")) == norm(gt.get(f)))
            csv_f.write(f'{r["id"]},{s["parse_ok"]},{s["schema_ok"]},{s["type_ok"]},'
                        f'{len(s["hallucinations"])},{len(s["leaks"])},'
                        f'"{s["missing"]}","{s["extra"]}"\n')
            print(f'{r["id"]}: parse={s["parse_ok"]} schema={s["schema_ok"]} '
                  f'halluc={s["hallucinations"]} leak={s["leaks"]}')

    print("\n===== ACT 1 FORMAT REPORT CARD =====")
    print(f"Parse rate        : {parse}/{n} = {100*parse/n:.1f}%   (target >=99%)")
    print(f"Schema adherence  : {schema}/{n} = {100*schema/n:.1f}%  (target >=99%)")
    print(f"List-type correct : {type_ok}/{n}")
    print(f"Hallucinations    : {halluc_total} total  (target ~0; Certifications excluded)")
    print(f"Field leaks       : {leak_total} total  (target 0)")
    if contact_total:
        print(f"[peek] Email+Phone exact: {contact_hits}/{contact_total} "
              f"({100*contact_hits/contact_total:.0f}%) — not the Act 1 grade, just a sanity sniff")
    passed = (parse/n >= 0.99 and schema/n >= 0.99 and halluc_total == 0 and leak_total == 0)
    print("\nVERDICT:", "PASS -> proceed to Act 2 🎉" if passed
          else "NOT YET -> add epochs (NOT lr), and/or fix offending synthetic examples. "
               "Inspect act1_raw_outputs.jsonl to see what it actually emitted.")


if __name__ == "__main__":
    main()