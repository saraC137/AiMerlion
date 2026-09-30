"""
prepare_training_data.py

FAIRY CODEMOTHER'S LLM TRAINING DATA GENERATOR

Bridges AiMerlion annotation tool -> Unsloth fine-tuning pipeline.
Reads from resume_extractions.db (4 tables) and generates
ShareGPT JSONL for fine-tuning.

Tables used:
  1. ner_documents        -> doc_id, candidate_id, status, function, industry
  2. raw_extractions      -> raw_text (full resume text)
  3. ner_annotations      -> entity spans (PERSON_NAME, PHONE, etc.)
  4. structured_extractions -> parsed fields (name, experience_json, etc.)

Usage:
    python prepare_training_data.py --db resume_extractions.db
    python prepare_training_data.py --db resume_extractions.db --status all
    python prepare_training_data.py --db resume_extractions.db --preview 3
"""

import os, sys, json, sqlite3, argparse, random
from datetime import datetime
from typing import List, Dict, Any, Optional

SYSTEM_PROMPT = (
    "You are a precise resume data extraction assistant specializing in "
    "Singapore and Malaysia resumes. Extract the requested information "
    "and return valid JSON only. Handle 8-digit phone numbers (+65/+60), "
    "date-first experience formats, local company names, and multilingual "
    "content accurately."
)

INSTRUCTION_TEMPLATE = (
    "Extract structured data from this resume. "
    "Return valid JSON with these fields where available: "
    "name, email, phone, date_of_birth, location, summary, "
    "hard_skills (array), soft_skills (array), "
    "experience (array of {{title, company, duration}}), "
    "education (array of {{degree, institution}}), "
    "certifications (array), languages (array), "
    "function (job function category), industry (industry category).\n\n"
    "Resume:\n{resume_text}"
)


def load_annotated_resumes(db_path: str, status_filter: str = "completed") -> List[Dict]:
    """Load annotated resume data by joining across AiMerlion tables."""
    if not os.path.exists(db_path):
        print(f"  Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if status_filter == "all":
        doc_rows = conn.execute(
            "SELECT doc_id, candidate_id, status, function, industry FROM ner_documents"
        ).fetchall()
    else:
        doc_rows = conn.execute(
            "SELECT doc_id, candidate_id, status, function, industry "
            "FROM ner_documents WHERE status = ?", (status_filter,)
        ).fetchall()

    if not doc_rows:
        print(f"  No documents with status='{status_filter}' found!")
        conn.close()
        return []

    results = []
    skipped = {"no_text": 0, "too_short": 0, "no_annotations": 0}

    for doc_row in doc_rows:
        doc_id = doc_row["doc_id"]
        candidate_id = doc_row["candidate_id"]

        # Get raw text from raw_extractions (primary source)
        raw_text_row = conn.execute(
            "SELECT raw_text FROM raw_extractions "
            "WHERE candidate_id = ? ORDER BY extraction_timestamp DESC LIMIT 1",
            (candidate_id,)
        ).fetchone()
        raw_text = raw_text_row["raw_text"] if raw_text_row else ""

        if not raw_text:
            skipped["no_text"] += 1
            continue
        if len(raw_text.strip()) < 100:
            skipped["too_short"] += 1
            continue

        # Get entity annotations (human-labeled spans)
        ann_rows = conn.execute(
            "SELECT entity_type, text_content FROM ner_annotations "
            "WHERE doc_id = ? AND layer = 0 ORDER BY char_start", (doc_id,)
        ).fetchall()

        annotations = {}
        for ann in ann_rows:
            etype = ann["entity_type"]
            text = (ann["text_content"] or "").strip()
            if text:
                if etype not in annotations:
                    annotations[etype] = []
                if text not in annotations[etype]:
                    annotations[etype].append(text)

        # Get structured extraction data
        structured = {}
        se_row = conn.execute(
            "SELECT name, email, phone, location, summary, "
            "experience_json, education_json, skills_json "
            "FROM structured_extractions WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()

        if se_row:
            structured = {
                "name": se_row["name"] or "",
                "email": se_row["email"] or "",
                "phone": se_row["phone"] or "",
                "location": se_row["location"] or "",
                "summary": se_row["summary"] or "",
            }
            for jf in ["experience_json", "education_json", "skills_json"]:
                try:
                    structured[jf] = json.loads(se_row[jf]) if se_row[jf] else []
                except (json.JSONDecodeError, TypeError):
                    structured[jf] = []

        if not annotations and not structured.get("name"):
            skipped["no_annotations"] += 1
            continue

        results.append({
            "doc_id": doc_id, "candidate_id": candidate_id,
            "raw_text": raw_text, "annotations": annotations,
            "structured": structured,
            "function": doc_row["function"] or "",
            "industry": doc_row["industry"] or "",
        })

    conn.close()

    if any(skipped.values()):
        total_skip = sum(skipped.values())
        print(f"  Skipped {total_skip} documents:")
        for reason, count in skipped.items():
            if count: print(f"      {count} - {reason}")

    return results


def build_golden_output(item: Dict) -> Dict:
    """Build the ideal JSON output merging annotations + structured data."""
    ann = item["annotations"]
    se = item["structured"]
    output = {}

    # Personal info (annotations > structured)
    output["name"] = ann.get("PERSON_NAME", [None])[0] or se.get("name", "")
    output["email"] = ann.get("EMAIL", [None])[0] or se.get("email", "")
    output["phone"] = ann.get("PHONE", [None])[0] or se.get("phone", "")

    loc = ann.get("LOCATION", [None])[0] or se.get("location", "")
    if loc: output["location"] = loc

    dob = ann.get("DATE_OF_BIRTH", [None])[0]
    if dob: output["date_of_birth"] = dob

    summary = se.get("summary", "")
    if summary and len(summary) > 20: output["summary"] = summary

    # Skills
    hard = ann.get("HARD_SKILL", []) or ann.get("SKILL", [])
    soft = ann.get("SOFT_SKILL", [])
    if not hard and not soft:
        se_skills = se.get("skills_json", [])
        if isinstance(se_skills, list):
            hard = [s for s in se_skills if isinstance(s, str)][:20]
        elif isinstance(se_skills, dict):
            hard = [str(v) for v in se_skills.values() if v][:20]
    if hard: output["hard_skills"] = hard
    if soft: output["soft_skills"] = soft

    # Experience
    experience = []
    titles = ann.get("JOB_TITLE", [])
    companies = ann.get("COMPANY", []) or ann.get("ORGANIZATION", [])
    durations = ann.get("WORK_DURATION", []) or ann.get("WORK_DATE", [])

    if titles or companies:
        for i in range(max(len(titles), len(companies))):
            e = {}
            if i < len(titles): e["title"] = titles[i]
            if i < len(companies): e["company"] = companies[i]
            if i < len(durations): e["duration"] = durations[i]
            if e: experience.append(e)
    else:
        se_exp = se.get("experience_json", [])
        if not isinstance(se_exp, list):
            se_exp = list(se_exp.values()) if isinstance(se_exp, dict) else []
        for exp in se_exp[:10]:
            if isinstance(exp, dict):
                e = {}
                if exp.get("role") or exp.get("title"):
                    e["title"] = exp.get("role") or exp.get("title", "")
                if exp.get("company"): e["company"] = exp["company"]
                if exp.get("dates") or exp.get("duration"):
                    e["duration"] = exp.get("dates") or exp.get("duration", "")
                if e: experience.append(e)
    if experience: output["experience"] = experience

    # Education
    education = []
    degrees = ann.get("DEGREE", [])
    institutions = ann.get("INSTITUTION", [])

    if degrees or institutions:
        for i in range(max(len(degrees), len(institutions))):
            e = {}
            if i < len(degrees): e["degree"] = degrees[i]
            if i < len(institutions): e["institution"] = institutions[i]
            if e: education.append(e)
    else:
        se_edu = se.get("education_json", [])
        if not isinstance(se_edu, list):
            se_edu = list(se_edu.values()) if isinstance(se_edu, dict) else []
        for edu in se_edu[:5]:
            if isinstance(edu, dict):
                e = {}
                if edu.get("degree"): e["degree"] = edu["degree"]
                if edu.get("institution"): e["institution"] = edu["institution"]
                if e: education.append(e)
    if education: output["education"] = education

    # Other fields
    certs = ann.get("CERTIFICATION", [])
    if certs: output["certifications"] = certs
    langs = ann.get("LANGUAGE", []) or ann.get("LANGUAGE_SKILL", [])
    if langs: output["languages"] = langs
    if item.get("function"): output["function"] = item["function"]
    if item.get("industry"): output["industry"] = item["industry"]

    return output


def format_as_sharegpt(item: Dict, golden_output: Dict) -> Dict:
    """Format one training example as ShareGPT conversation for Unsloth."""
    resume_text = item["raw_text"][:4000]
    instruction = INSTRUCTION_TEMPLATE.format(resume_text=resume_text)
    response = json.dumps(golden_output, ensure_ascii=False, indent=None)
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": instruction},
            {"from": "gpt", "value": response},
        ]
    }


def print_quality_report(items, golden_outputs):
    """Print a quality report showing field coverage."""
    total = len(items)
    if not total: return

    print(f"\n  DATA QUALITY REPORT ({total} resumes)")
    print("  " + "-" * 50)

    fields = ["name","email","phone","location","hard_skills","soft_skills",
              "experience","education","certifications","languages","function","industry"]
    counts = {}
    for f in fields:
        counts[f] = sum(1 for o in golden_outputs
                       if o.get(f) and ((isinstance(o[f], str) and o[f].strip()) or
                                        (isinstance(o[f], list) and o[f])))

    for f in sorted(fields, key=lambda x: -counts[x]):
        pct = counts[f] / total * 100
        bar = "█" * int(pct/5) + "░" * (20 - int(pct/5))
        icon = "✅" if pct >= 60 else "⚠️ " if pct >= 30 else "❌"
        print(f"    {icon} {f:18s} {bar} {pct:5.1f}% ({counts[f]}/{total})")

    # Entity annotation counts
    entity_counts = {}
    for item in items:
        for etype, texts in item["annotations"].items():
            entity_counts[etype] = entity_counts.get(etype, 0) + len(texts)
    if entity_counts:
        print(f"\n  Entity annotations (total spans):")
        for et, c in sorted(entity_counts.items(), key=lambda x: -x[1])[:12]:
            print(f"    {et:22s} -> {c} spans")

    print()
    if total < 30:
        print("  ⚠️  < 30 examples: model may underperform. Annotate more!")
    elif total < 100:
        print("  💡 50-100 examples is viable for QLoRA fine-tuning")
    else:
        print("  ✅ 100+ examples: excellent training set!")
    print("  " + "-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Generate LLM fine-tuning data from AiMerlion annotations")
    parser.add_argument("--db", default="resume_extractions.db", help="Database path")
    parser.add_argument("--output", default="train_data.jsonl", help="Output JSONL path")
    parser.add_argument("--status", default="completed", choices=["completed","in_progress","all"])
    parser.add_argument("--split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", type=int, default=0, help="Preview N examples without writing")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  💅✨ LLM TRAINING DATA GENERATOR ✨💅")
    print("  Annotations -> ShareGPT JSONL -> Unsloth")
    print("=" * 60)
    print(f"  Database:   {args.db}")
    print(f"  Output:     {args.output}")
    print(f"  Status:     {args.status}")
    print(f"  Val split:  {args.split*100:.0f}%")
    print("=" * 60)

    # Load
    print("\n  Loading annotated resumes...")
    items = load_annotated_resumes(args.db, args.status)
    if not items:
        print("  No usable annotated documents found!")
        sys.exit(1)
    print(f"  ✅ Found {len(items)} usable annotated resumes")

    # Build golden outputs
    print("  Building golden extraction outputs...")
    golden_outputs = [build_golden_output(item) for item in items]
    print_quality_report(items, golden_outputs)

    # Preview mode
    if args.preview > 0:
        print(f"\n  PREVIEW (first {args.preview} examples):")
        for i in range(min(args.preview, len(items))):
            fmt = format_as_sharegpt(items[i], golden_outputs[i])
            print(f"\n  --- Example {i+1} (doc: {items[i]['doc_id']}) ---")
            pretty = json.dumps(json.loads(fmt["conversations"][2]["value"]), indent=2, ensure_ascii=False)
            for line in pretty.split("\n"): print(f"    {line}")
        print("\n  Remove --preview to write files")
        return

    # Format + split
    print("\n  Formatting as ShareGPT conversations...")
    formatted = [format_as_sharegpt(i, o) for i, o in zip(items, golden_outputs)]
    random.seed(args.seed)
    random.shuffle(formatted)

    split_idx = max(1, int(len(formatted) * (1 - args.split)))
    train_data, val_data = formatted[:split_idx], formatted[split_idx:]

    # Write files
    with open(args.output, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✅ Training:   {len(train_data)} examples -> {args.output} ({os.path.getsize(args.output)/1024:.1f} KB)")

    val_path = args.output.replace(".jsonl", "_val.jsonl")
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✅ Validation:  {len(val_data)} examples -> {val_path} ({os.path.getsize(val_path)/1024:.1f} KB)")

    print("\n" + "=" * 60)
    print("  🎯 NEXT STEPS")
    print("=" * 60)
    print(f"  1. Review:  python prepare_training_data.py --db {args.db} --preview 3")
    print(f"  2. Train:   python finetune_unsloth.py --train-file {args.output} --val-file {val_path}")
    print(f"  3. Deploy:  ollama create aimerlion-resume -f resume_model_finetuned/Modelfile")
    print("=" * 60)
    print("  💅 Your golden training data is READY! 💅\n")

if __name__ == "__main__":
    main()