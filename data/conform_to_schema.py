"""
conform_to_schema.py  -  Force any resume record onto the CONFIRMED schema.

Confirmed extraction schema (17 fields the model learns to output):
  Name, Phone, Email, Current Location, Current Company, Current Title,
  Summary, Function, Industry(list), Language Skills(list),
  Certifications(list), hard_skills/tags(list), soft_skills/skills(list),
  Achievements(list), Work Experience(list), Project Experience(list),
  Education(list)

Nested entry shapes (uniform):
  Work Experience -> {company, title, from, to, responsibility(list)}
  Education       -> {school, major, degree, dates}

Handles BOTH input dialects:
  - golden set : snake_case keys, function/industry at TOP level, language_skills
                 as a string, hard_skills/soft_skills split, no Certifications.
  - synthetic  : Title-Case keys, already close to target.

Admin/CRM fields (ID, Page, Team, Gender, ...) are intentionally NOT emitted
in the extraction output -- your system fills those.

Dropped vs the golden set (by design of the confirmed schema; say the word to
keep any): per-job `location` in Work Experience, `gpa` in Education,
`linkedin`.

Usage:
    python conform_to_schema.py --in golden_sg_my.jsonl   --out golden_sg_my.conformed.jsonl
    python conform_to_schema.py --in synthetic_lite.jsonl --out synthetic_lite.conformed.jsonl
"""

import argparse, json, re

EXTRACT_FIELDS = [
    "Name", "Phone", "Email", "Current Location", "Current Company", "Current Title",
    "Summary", "Function", "Industry", "Language Skills",
    "Certifications", "hard_skills/tags", "soft_skills/skills", "Achievements",
    "Work Experience", "Project Experience", "Education",
]
LIST_FIELDS = {"Industry", "Language Skills", "Certifications", "hard_skills/tags",
               "soft_skills/skills", "Achievements", "Work Experience",
               "Project Experience", "Education"}

DASH = re.compile(r"\s*[-–—]\s*|\s+to\s+", re.I)
SPLITTERS = re.compile(r"\s*[,/;]\s*")


def pick(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def as_list(v):
    """Coerce a value to a list. Strings get split on , / ;  -> [] if empty."""
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [x for x in v if x not in (None, "")]
    if isinstance(v, str):
        return [s.strip() for s in SPLITTERS.split(v) if s.strip()]
    return [v]


def norm_experience(entries):
    out = []
    for e in as_list(entries):
        if not isinstance(e, dict):
            continue
        frm, to = pick(e, "from", default=""), pick(e, "to", default="")
        if not frm and not to:
            dates = pick(e, "dates", default="")
            if dates:
                parts = DASH.split(dates, maxsplit=1)
                frm = parts[0].strip()
                to = parts[1].strip() if len(parts) > 1 else ""
        resp = pick(e, "responsibility", "responsibilities", default=[])
        if isinstance(resp, str):
            resp = [resp] if resp else []
        out.append({
            "company": pick(e, "company", "Company", default=""),
            "title": pick(e, "title", "Title", default=""),
            "from": frm, "to": to,
            "responsibility": [r for r in resp if r],
        })
    return out


def norm_education(entries):
    out = []
    for e in as_list(entries):
        if not isinstance(e, dict):
            continue
        dates = pick(e, "dates", default="")
        if not dates:
            f, t = pick(e, "from", default=""), pick(e, "to", default="")
            dates = f"{f} - {t}".strip(" -") if (f or t) else ""
        out.append({
            "school": pick(e, "school", "institution", "School", default=""),
            "major": pick(e, "major", "field_of_study", default=""),
            "degree": pick(e, "degree", "Degree", default=""),
            "dates": dates,
        })
    return out


def to_schema(gt, top_function="", top_industry=""):
    r = {}
    r["Name"] = pick(gt, "Name", "name", default="")
    r["Phone"] = pick(gt, "Phone", "phone", default="")
    r["Email"] = pick(gt, "Email", "email", default="")
    r["Current Location"] = pick(gt, "Current Location", "current_location", default="")
    r["Current Company"] = pick(gt, "Current Company", "current_company", default="")
    r["Current Title"] = pick(gt, "Current Title", "current_title", default="")
    r["Summary"] = pick(gt, "Summary", "summary", default="")
    r["Function"] = pick(gt, "Function", "function", default="") or (top_function or "")
    r["Industry"] = as_list(pick(gt, "Industry", "industry", default=None)) or as_list(top_industry)
    r["Language Skills"] = as_list(pick(gt, "Language Skills", "language_skills", default=None))
    r["Certifications"] = as_list(pick(gt, "Certifications", "certifications", default=None))
    r["hard_skills/tags"] = as_list(pick(gt, "hard_skills/tags", "hard_skills", default=None))
    r["soft_skills/skills"] = as_list(pick(gt, "soft_skills/skills", "soft_skills", default=None))
    r["Achievements"] = as_list(pick(gt, "Achievements", "achievements", default=None))
    r["Work Experience"] = norm_experience(pick(gt, "Work Experience", "work_experience", default=[]))
    r["Project Experience"] = as_list(pick(gt, "Project Experience", "project_experience", default=[]))
    r["Education"] = norm_education(pick(gt, "Education", "education", default=[]))
    return {k: r[k] for k in EXTRACT_FIELDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    n = 0
    with open(args.inp, encoding="utf-8") as fi, open(args.out, "w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gt = rec.get("ground_truth", {})
            out_rec = {
                "id": rec.get("id"),
                "raw_text": rec.get("raw_text", ""),
                "ethnicity_tag": rec.get("ethnicity_tag"),
                "ground_truth": to_schema(gt,
                                          top_function=rec.get("function", ""),
                                          top_industry=rec.get("industry", "")),
            }
            fo.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"conformed {n} records -> {args.out}")


if __name__ == "__main__":
    main()