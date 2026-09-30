"""
Review-first resume extraction for the CRM edit profile form.

This is deliberately small and dependency-light:
- pasted text works with only the standard library
- DOCX extraction uses zip/xml from the standard library
- PDF extraction uses pypdf or pdfplumber only if already installed
- model extraction uses a local Ollama endpoint when available

Usage:
    python resume_crm_workflow.py --text-file resume.txt --out review.json
    python resume_crm_workflow.py --file resume.pdf --out review.json --autofill-js fill_profile.js
    python resume_crm_workflow.py --self-check
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


OLLAMA_URL = os.getenv("CRM_EXTRACT_OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
DEFAULT_MODEL = os.getenv("CRM_EXTRACT_MODEL", "merlion:latest")

REQUIRED_FIELDS = ["name", "industry", "function", "current_location", "phone", "email"]
REVIEW_FIELDS = {
    "date_of_birth",
    "gender",
    "marital_status",
    "annual_salary",
    "industry",
    "function",
}

PROFILE_TEMPLATE = {
    "type": "Candidate",
    "name": "",
    "industry": "",
    "function": "",
    "current_location": "",
    "phone": "",
    "email": "",
    "expected_location": "",
    "date_of_birth": "",
    "gender": "",
    "first_degree": "",
    "annual_salary": "",
    "whatsapp": "",
    "wechat": "",
    "anmelden": "",
    "marital_status": "",
    "source": "",
    "linkedin": "",
    "tags": [],
    "summary": "",
    "languages": [{"language": "", "proficiency": ""}],
    "work_experience": [
        {
            "from": "",
            "to": "",
            "current_job": False,
            "company": "",
            "title": "",
            "work_location": "",
            "responsibility": "",
        }
    ],
    "education": [
        {
            "from": "",
            "to": "",
            "school": "",
            "degree": "",
            "major": "",
            "is_985": False,
            "is_211": False,
            "is_qs200": False,
            "is_ivy": False,
        }
    ],
    "project_experience": "",
    "creator": "",
}


PROMPT = """You are a resume extraction model for Singapore and Malaysia recruiting.
Extract only facts explicitly present in the resume.

Return ONLY one JSON object matching this schema:
{
  "type": "Candidate",
  "name": "",
  "industry": "",
  "function": "",
  "current_location": "",
  "phone": "",
  "email": "",
  "expected_location": "",
  "date_of_birth": "",
  "gender": "",
  "first_degree": "",
  "annual_salary": "",
  "whatsapp": "",
  "wechat": "",
  "anmelden": "",
  "marital_status": "",
  "source": "",
  "linkedin": "",
  "tags": [],
  "summary": "",
  "languages": [{"language": "", "proficiency": ""}],
  "work_experience": [{"from": "", "to": "", "current_job": false, "company": "", "title": "", "work_location": "", "responsibility": ""}],
  "education": [{"from": "", "to": "", "school": "", "degree": "", "major": "", "is_985": false, "is_211": false, "is_qs200": false, "is_ivy": false}],
  "project_experience": "",
  "creator": ""
}

Rules:
- Preserve Singapore/Malaysia terms exactly: Pte Ltd, Sdn Bhd, Berhad, CMFAS, MAS, GCE O Level, GCE N Level, ITE, NITEC, Polytechnic, bin/binti, s/o, d/o.
- Fill gender, date_of_birth, marital_status, and annual_salary only when explicitly stated.
- Do not infer missing required fields.
- Use empty strings/lists for absent values.
- Set current_job true when the work item says Present, Current, now, or is clearly the current role.
"""


def blank_profile() -> dict[str, Any]:
    return copy.deepcopy(PROFILE_TEMPLATE)


def looks_letter_spaced(text: str) -> bool:
    """True when most tokens are single characters, e.g. 'C o n t a c t'.

    Designer/Canva-style PDFs position each glyph individually, so extractors
    emit one space between every letter and two+ spaces between words.
    """
    tokens = text.split()
    if len(tokens) < 30:
        return False
    return sum(1 for t in tokens if len(t) == 1) / len(tokens) > 0.5


def despace_text(text: str) -> str:
    """Repair letter-spaced text: drop intra-word spaces, keep word breaks.

    Relies on the convention above (1 space inside a word, 2+ between words),
    so it must run before any pass that collapses runs of spaces.
    """
    if not looks_letter_spaced(text):
        return text
    out = []
    for line in text.split("\n"):
        line = re.sub(r" {2,}", "\x00", line)  # protect word boundaries
        line = line.replace(" ", "")           # drop intra-word spaces
        out.append(line.replace("\x00", " "))
    return "\n".join(out)


def normalize_text(text: str) -> str:
    replacements = {
        "â€“": "-",
        "â€”": "-",
        "â€˜": "'",
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "â€¢": "-",
        "Â": "",
        "\u00a0": " ",
        "\uf0b7": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = unicodedata.normalize("NFKC", text)
    text = despace_text(text)  # before the space-collapse below
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".text"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".docx":
        return read_docx(path)
    if suffix == ".pdf":
        return read_pdf(path)
    raise ValueError(f"Unsupported file type: {path.suffix}. Use .txt, .pdf, or .docx.")


def read_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as docx:
        names = ["word/document.xml"] + sorted(
            n for n in docx.namelist() if n.startswith("word/header") or n.startswith("word/footer")
        )
        chunks: list[str] = []
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        for name in names:
            if name not in docx.namelist():
                continue
            root = ElementTree.fromstring(docx.read(name))
            for paragraph in root.findall(".//w:p", ns):
                text = "".join(node.text or "" for node in paragraph.findall(".//w:t", ns)).strip()
                if text:
                    chunks.append(text)
        return "\n".join(chunks)


def read_pdf(path: Path) -> str:
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        pass

    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(str(path)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as exc:
        raise RuntimeError(
            "Could not extract PDF text. Install/use an environment with pypdf or pdfplumber, "
            "or paste the resume text with --text-file."
        ) from exc


def extract_with_ollama(text: str, model: str, timeout: int) -> dict[str, Any] | None:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    content = body.get("message", {}).get("content", "")
    return parse_json_object(content)


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = re.sub(r"```(?:json)?", "", text or "").replace("```", "")
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def heuristic_extract(text: str) -> dict[str, Any]:
    profile = blank_profile()
    profile["email"] = first_match(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    profile["linkedin"] = first_match(r"https?://(?:www\.)?linkedin\.com/[^\s)>,]+", text)
    profile["phone"] = normalize_phone(first_match(r"(?:\+?65[\s-]?)?[689]\d{3}[\s-]?\d{4}|(?:\+?60[\s-]?)?1\d[\s-]?\d{3}[\s-]?\d{4}", text))
    profile["name"] = guess_name(text)
    profile["current_location"] = guess_location(text)
    profile["date_of_birth"] = first_match(
        r"(?:date of birth|dob|birth date)\s*[:\-]?\s*([0-3]?\d[ /.-][A-Za-z]{3,9}[ /.-]\d{2,4}|[0-3]?\d[ /.-][01]?\d[ /.-]\d{2,4})",
        text,
        group=1,
        flags=re.I,
    )
    profile["annual_salary"] = first_match(
        r"(?:annual salary|expected salary|current salary|salary)\s*[:\-]?\s*((?:SGD|S\$|\$|RM|MYR)?\s*\d[\d,]*(?:\s*(?:k|K))?)",
        text,
        group=1,
        flags=re.I,
    )
    profile["marital_status"] = first_match(r"(?:marital status)\s*[:\-]?\s*(single|married|divorced|widowed)", text, group=1, flags=re.I)
    profile["gender"] = first_match(r"(?:gender|sex)\s*[:\-]?\s*(male|female)", text, group=1, flags=re.I).title()
    profile["languages"] = guess_languages(text)
    profile["summary"] = section_after(text, ["summary", "profile", "career objective"], ["experience", "employment", "education", "skills"])
    profile["education"] = guess_education(text)
    profile["work_experience"] = guess_work_experience(text)
    profile["first_degree"] = first_degree(profile["education"])
    return profile


def first_match(pattern: str, text: str, group: int = 0, flags: int = 0) -> str:
    match = re.search(pattern, text, flags)
    return match.group(group).strip() if match else ""


def guess_name(text: str) -> str:
    for line in text.splitlines()[:10]:
        line = line.strip(" -|\t")
        if not line or "@" in line or re.search(r"\d", line):
            continue
        if re.search(r"\b(resume|curriculum vitae|cv|profile|email|phone|mobile)\b", line, re.I):
            continue
        if len(line.split()) <= 7:
            return line
    return ""


def guess_location(text: str) -> str:
    for pattern in [
        r"((?:Blk|Block)\s+.+?Singapore\s+\d{6})",
        r"([A-Za-z0-9 #,\-]+Singapore\s+\d{6})",
        r"(Singapore)",
        r"(Kuala Lumpur|Selangor|Johor Bahru|Penang|Petaling Jaya|Malaysia)",
    ]:
        value = first_match(pattern, text, flags=re.I)
        if value:
            return re.sub(r"\s+", " ", value).strip(" ,")
    return ""


def normalize_phone(phone: str) -> str:
    if not phone:
        return ""
    digits = re.sub(r"\D", "", phone)
    if len(digits) == 8 and digits[0] in "689":
        return f"+65 {digits[:4]} {digits[4:]}"
    if len(digits) == 10 and digits.startswith("65"):
        return f"+65 {digits[2:6]} {digits[6:]}"
    if digits.startswith("60") and len(digits) >= 10:
        return "+" + digits
    return phone.strip()


def guess_languages(text: str) -> list[dict[str, str]]:
    line = first_match(r"(?:languages?|language skills?)\s*[:\-]?\s*(.+)", text, group=1, flags=re.I)
    if not line:
        return []
    found = []
    for language in ["English", "Mandarin", "Chinese", "Malay", "Bahasa Malaysia", "Tamil", "Hokkien", "Cantonese"]:
        if re.search(rf"\b{re.escape(language)}\b", line, re.I):
            found.append({"language": language, "proficiency": ""})
    return dedupe_dicts(found, "language")


def section_after(text: str, starts: list[str], stops: list[str]) -> str:
    start_re = "|".join(re.escape(s) for s in starts)
    stop_re = "|".join(re.escape(s) for s in stops)
    match = re.search(rf"(?im)^\s*(?:{start_re})\s*:?\s*$([\s\S]*?)(?=^\s*(?:{stop_re})\b|\Z)", text)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip(" -")


def guess_education(text: str) -> list[dict[str, Any]]:
    edu_text = section_after(text, ["education", "academic qualifications"], ["experience", "employment", "skills", "certifications", "projects"])
    if not edu_text:
        return []
    schools = [
        "National University of Singapore",
        "Nanyang Technological University",
        "Singapore Management University",
        "Singapore Polytechnic",
        "Ngee Ann Polytechnic",
        "Temasek Polytechnic",
        "Nanyang Polytechnic",
        "Republic Polytechnic",
        "ITE",
    ]
    rows = []
    for line in edu_text.splitlines():
        clean = line.strip(" -|")
        if not clean:
            continue
        school = next((s for s in schools if s.lower() in clean.lower()), "")
        degree = first_match(r"\b((?:Bachelor|Master|Diploma|Higher NITEC|NITEC|GCE [ON]'? Level|GCE O Level|GCE N Level)[^|,\n]*)", clean, flags=re.I)
        dates = first_match(r"((?:19|20)\d{2}\s*(?:-|to|–|~)\s*(?:(?:19|20)\d{2}|present|current)|(?:19|20)\d{2})", clean, flags=re.I)
        if school or degree:
            rows.append(normalize_education({"school": school or clean, "degree": degree, "dates": dates}))
    return rows


def guess_work_experience(text: str) -> list[dict[str, Any]]:
    exp_text = section_after(text, ["experience", "work experience", "employment history", "career history"], ["education", "skills", "certifications", "projects"])
    if not exp_text:
        return []
    rows = []
    company_pattern = re.compile(r"\b([A-Z][A-Za-z0-9&()., '\-/]+(?:Pte Ltd|Sdn Bhd|Berhad|Limited|Ltd|Corporation|Bank|Ministry)[A-Za-z0-9&()., '\-/]*)", re.I)
    lines = [line.strip(" -*\t") for line in exp_text.splitlines() if line.strip()]
    for i, line in enumerate(lines[:40]):
        company = first_match(company_pattern.pattern, line, flags=re.I)
        if not company:
            continue
        nearby = " ".join(lines[i : i + 3])
        dates = first_match(r"((?:[A-Za-z]{3,9}\s+)?(?:19|20)\d{2}\s*(?:-|to|–|~)\s*(?:(?:[A-Za-z]{3,9}\s+)?(?:19|20)\d{2}|present|current))", nearby, flags=re.I)
        title = first_match(r"(?:role|title|position)\s*[:\-]\s*([^|,]+)", nearby, group=1, flags=re.I)
        item = normalize_work({"company": company, "title": title, "dates": dates})
        rows.append(item)
    return rows


def coerce_profile(data: dict[str, Any]) -> dict[str, Any]:
    profile = blank_profile()
    source = data.get("ground_truth", data)

    aliases = {
        "name": ["name", "Name"],
        "industry": ["industry", "Industry"],
        "function": ["function", "Function"],
        "current_location": ["current_location", "Current Location", "location"],
        "phone": ["phone", "Phone"],
        "email": ["email", "Email"],
        "expected_location": ["expected_location", "Expected Location"],
        "date_of_birth": ["date_of_birth", "Date of Birth", "dob"],
        "gender": ["gender", "Gender"],
        "annual_salary": ["annual_salary", "Annual Salary", "salary"],
        "summary": ["summary", "Summary"],
        "linkedin": ["linkedin", "LinkedIn"],
        "tags": ["tags", "Tags", "hard_skills/tags", "hard_skills", "Hard Skills"],
        "project_experience": ["project_experience", "Project Experience"],
    }
    for target, names in aliases.items():
        value = pick(source, names)
        if value not in (None, ""):
            profile[target] = value

    profile["type"] = pick(source, ["type", "Type"]) or "Candidate"
    profile["languages"] = normalize_languages(pick(source, ["languages", "Language Skills", "language_skills"]))
    profile["work_experience"] = normalize_work_list(pick(source, ["work_experience", "Work Experience"]))
    profile["education"] = normalize_education_list(pick(source, ["education", "Education"]))
    profile["phone"] = normalize_phone(str(profile["phone"]))
    profile["first_degree"] = profile["first_degree"] or first_degree(profile["education"])
    profile["tags"] = as_list(profile["tags"])
    return remove_empty_placeholder_rows(profile)


PLACEHOLDER_RE = re.compile(
    r"^(?:n/?a|nil|none|na|not\s+(?:provided|specified|available|applicable|stated|"
    r"mentioned|disclosed|indicated)|unknown|unspecified|undisclosed|negotiable|"
    r"tbd|tba|[-.\s]+)$",
    re.I,
)
PLACEHOLDER_PREFIX_RE = re.compile(r"^to\s+be\s+(?:discussed|confirmed|advised|negotiated)\b", re.I)


def is_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip().rstrip(".").strip()
    return bool(PLACEHOLDER_RE.match(stripped) or PLACEHOLDER_PREFIX_RE.match(stripped))


def pick(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in data and data[key] not in (None, "") and not is_placeholder(data[key]):
            return data[key]
    return ""


def as_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [v for v in value if v not in (None, "")]
    if isinstance(value, str):
        return [v.strip() for v in re.split(r"[,;/]\s*", value) if v.strip()]
    return [value]


def normalize_languages(value: Any) -> list[dict[str, str]]:
    rows = []
    for item in as_list(value):
        if isinstance(item, dict):
            language = str(pick(item, ["language", "Language"]) or "").strip()
            proficiency = str(pick(item, ["proficiency", "level", "Proficiency"]) or "").strip()
        else:
            language, proficiency = str(item).strip(), ""
        if language:
            rows.append({"language": language, "proficiency": proficiency})
    return dedupe_dicts(rows, "language")


def normalize_work_list(value: Any) -> list[dict[str, Any]]:
    rows = [normalize_work(item) for item in as_list(value) if isinstance(item, dict)]
    for i, row in enumerate(rows):
        row["current_job"] = bool(row["current_job"] or (i == 0 and row["to"].lower() in {"", "present", "current", "now"}))
    return rows


def normalize_work(item: dict[str, Any]) -> dict[str, Any]:
    frm = str(pick(item, ["from", "From"]) or "").strip()
    to = str(pick(item, ["to", "To"]) or "").strip()
    dates = str(pick(item, ["dates", "date", "period"]) or "").strip()
    if dates and not (frm or to):
        frm, to = split_dates(dates)
    responsibility = pick(item, ["responsibility", "responsibilities", "details", "Responsibility"]) or ""
    if isinstance(responsibility, list):
        responsibility = "\n".join(str(r).strip() for r in responsibility if r)
    return {
        "from": frm,
        "to": to,
        "current_job": bool(pick(item, ["current_job", "Current Job"])),
        "company": str(pick(item, ["company", "Company"]) or "").strip(),
        "title": str(pick(item, ["title", "Title"]) or "").strip(),
        "work_location": str(pick(item, ["work_location", "location", "Work location"]) or "").strip(),
        "responsibility": str(responsibility).strip(),
    }


def normalize_education_list(value: Any) -> list[dict[str, Any]]:
    return [normalize_education(item) for item in as_list(value) if isinstance(item, dict)]


def normalize_education(item: dict[str, Any]) -> dict[str, Any]:
    frm = str(pick(item, ["from", "From"]) or "").strip()
    to = str(pick(item, ["to", "To"]) or "").strip()
    dates = str(pick(item, ["dates", "date", "period"]) or "").strip()
    if dates and not (frm or to):
        frm, to = split_dates(dates)
    degree = pick(item, ["degree", "Degree", "qualification", "Qualification"]) or ""
    if isinstance(degree, list):
        degree = ", ".join(str(d) for d in degree if d)
    return {
        "from": frm,
        "to": to,
        "school": str(pick(item, ["school", "School", "institution"]) or "").strip(),
        "degree": str(degree).strip(),
        "major": str(pick(item, ["major", "Major in", "field_of_study"]) or "").strip(),
        "is_985": bool(pick(item, ["is_985", "985"])),
        "is_211": bool(pick(item, ["is_211", "211"])),
        "is_qs200": bool(pick(item, ["is_qs200", "QS200"])),
        "is_ivy": bool(pick(item, ["is_ivy", "IVY"])),
    }


def split_dates(dates: str) -> tuple[str, str]:
    parts = re.split(r"\s*(?:-|–|—|~|\bto\b)\s*", dates, maxsplit=1, flags=re.I)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return dates.strip(), ""


def first_degree(education: list[dict[str, Any]]) -> str:
    for item in education:
        degree = item.get("degree", "")
        if re.search(r"\b(bachelor|master|phd|degree)\b", degree, re.I):
            return degree
    return education[0].get("degree", "") if education else ""


def remove_empty_placeholder_rows(profile: dict[str, Any]) -> dict[str, Any]:
    profile["languages"] = [r for r in profile["languages"] if r.get("language")]
    profile["work_experience"] = [
        r for r in profile["work_experience"] if r.get("company") or r.get("title") or r.get("responsibility")
    ]
    profile["education"] = [r for r in profile["education"] if r.get("school") or r.get("degree")]
    return profile


def dedupe_dicts(rows: list[dict[str, str]], key: str) -> list[dict[str, str]]:
    seen = set()
    out = []
    for row in rows:
        marker = row.get(key, "").strip().lower()
        if marker and marker not in seen:
            out.append(row)
            seen.add(marker)
    return out


def build_review(profile: dict[str, Any], text: str) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if not profile.get(field)]
    for i, item in enumerate(profile.get("work_experience", []), start=1):
        if not item.get("company"):
            missing.append(f"work_experience[{i}].company")
        if not item.get("title"):
            missing.append(f"work_experience[{i}].title")

    needs_review = sorted(field for field in REVIEW_FIELDS if profile.get(field))
    return {
        "profile": profile,
        "missing_required": missing,
        "needs_review": needs_review,
        "source_spans": source_spans(profile, text),
    }


def source_spans(profile: dict[str, Any], text: str) -> dict[str, str]:
    spans = {}
    for field in ["name", "phone", "email", "current_location", "linkedin"]:
        value = str(profile.get(field, "")).strip()
        if value and value.lower() in text.lower():
            spans[field] = value
    return spans


def extract_review(text: str, model: str, timeout: int, no_model: bool = False) -> dict[str, Any]:
    normalized = normalize_text(text)
    model_data = None if no_model else extract_with_ollama(normalized, model, timeout)
    profile = coerce_profile(model_data) if model_data else heuristic_extract(normalized)
    return build_review(profile, normalized)


def find_candidate_folders(base: Path) -> list[Path]:
    """Discover candidate folders under merlion_resumes/.

    Mirrors core/main.py:find_candidate_folders — handles both a flat layout
    (merlion_resumes/100000_Name/) and a date-wrapped one
    (merlion_resumes/2025-07-10_16-54-36/100000_Name/).
    """
    folders: list[Path] = []
    for item in sorted(base.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        if re.match(r"\d{4}-\d{2}-\d{2}", item.name):
            folders += [c for c in sorted(item.iterdir()) if c.is_dir() and not c.name.startswith(".")]
        elif re.match(r"^\d+", item.name):
            folders.append(item)
    return folders


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".text"}


def find_resume_file(folder: Path) -> Path | None:
    """First readable resume file in a candidate folder (pdf/docx/txt)."""
    files = [
        f for f in sorted(folder.iterdir())
        if f.is_file() and not f.name.startswith("~$") and f.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return files[0] if files else None


def run_batch(args: argparse.Namespace) -> int:
    base = args.dir
    if not base.is_dir():
        raise SystemExit(f"--dir not found: {base}")
    out_dir = args.out_dir or (base.parent / "crm_reviews")
    out_dir.mkdir(parents=True, exist_ok=True)

    folders = find_candidate_folders(base)
    if args.limit:
        folders = folders[: args.limit]
    print(f"Found {len(folders)} candidate folders under {base}")

    summary = []
    for i, folder in enumerate(folders, 1):
        out_path = out_dir / f"{folder.name}.json"
        if out_path.exists() and not args.overwrite:
            summary.append({"candidate": folder.name, "status": "skipped (exists)"})
            continue
        resume = find_resume_file(folder)
        if resume is None:
            unsupported = [f.suffix.lower() for f in folder.iterdir() if f.is_file()]
            print(f"  [{i}/{len(folders)}] {folder.name}: no readable resume ({unsupported})")
            summary.append({"candidate": folder.name, "status": f"unsupported {unsupported}"})
            continue
        try:
            raw = read_text(resume)
            review = extract_review(raw, model=args.model, timeout=args.timeout, no_model=args.no_model)
        except Exception as exc:  # one bad file must not kill the batch
            print(f"  [{i}/{len(folders)}] {folder.name}: ERROR {exc}")
            summary.append({"candidate": folder.name, "file": resume.name, "status": f"error: {exc}"})
            continue
        out_path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        profile = review["profile"]
        summary.append({
            "candidate": folder.name,
            "file": resume.name,
            "name": profile.get("name", ""),
            "missing_required": review["missing_required"],
            "status": "ok",
        })
        print(f"  [{i}/{len(folders)}] {folder.name}: {profile.get('name') or '(no name)'} "
              f"-> {out_path.name}  missing={review['missing_required'] or 'none'}")

    summary_path = out_dir / "_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for s in summary if s["status"] == "ok")
    print(f"\nDone: {ok} extracted, {len(summary) - ok} skipped/failed.")
    print(f"Per-candidate JSON in {out_dir}\\  (summary: {summary_path.name})")
    return 0


def build_autofill_js(review: dict[str, Any]) -> str:
    profile = review["profile"]
    payload = json.dumps(profile, ensure_ascii=False, indent=2)
    return f"""// Paste this into the CRM profile page console after reviewing the JSON.
// ponytail: label-based autofill is intentionally simple; if custom widgets fail, upgrade to a selector map.
(() => {{
  const profile = {payload};
  const fire = (el) => {{
    el.dispatchEvent(new Event('input', {{ bubbles: true }}));
    el.dispatchEvent(new Event('change', {{ bubbles: true }}));
  }};
  const norm = (s) => String(s || '').replace(/[*:]/g, '').replace(/\\s+/g, ' ').trim().toLowerCase();
  const fields = () => Array.from(document.querySelectorAll('input, textarea, select'))
    .filter(el => !el.disabled && el.type !== 'hidden');
  const textNodes = () => Array.from(document.querySelectorAll('label, td, th, span, div'))
    .filter(el => el.children.length <= 3 && norm(el.textContent));
  const byLabel = (label) => {{
    const wanted = norm(label);
    const labelEl = textNodes().find(el => norm(el.textContent) === wanted || norm(el.textContent).startsWith(wanted + ' '));
    if (!labelEl) return null;
    const scopes = [labelEl.parentElement, labelEl.parentElement?.parentElement, labelEl.closest('tr'), labelEl.closest('form'), document];
    for (const scope of scopes.filter(Boolean)) {{
      const el = Array.from(scope.querySelectorAll('input, textarea, select'))
        .find(el => !el.disabled && el.type !== 'hidden' && el.offsetParent !== null);
      if (el) return el;
    }}
    return null;
  }};
  const byName = (names) => fields().find(el => names.some(name => norm(el.name).includes(name) || norm(el.id).includes(name)));
  const set = (labels, value, names = []) => {{
    if (value === undefined || value === null || value === '') return false;
    const el = labels.map(byLabel).find(Boolean) || byName(names);
    if (!el) return false;
    if (el.tagName === 'SELECT') {{
      const option = Array.from(el.options).find(o => norm(o.text) === norm(value) || norm(o.value) === norm(value));
      if (option) el.value = option.value;
      else return false;
    }} else {{
      el.value = Array.isArray(value) ? value.join(', ') : String(value);
    }}
    fire(el);
    return true;
  }};
  const clickRadio = (label, value) => {{
    if (!value) return false;
    const radios = Array.from(document.querySelectorAll('input[type=radio]'));
    const radio = radios.find(r => norm(r.value) === norm(value) || norm(r.parentElement?.textContent).includes(norm(value)));
    if (radio) {{ radio.click(); return true; }}
    return set([label], value);
  }};
  const misses = [];
  const trySet = (labels, value, names) => {{ if (!set(labels, value, names)) misses.push(labels[0]); }};

  clickRadio('Type', profile.type || 'Candidate');
  clickRadio('Gender', profile.gender);
  trySet(['Name'], profile.name, ['name']);
  trySet(['Industry'], profile.industry, ['industry']);
  trySet(['Function'], profile.function, ['function']);
  trySet(['Current Location'], profile.current_location, ['current_location', 'location']);
  trySet(['Phone'], profile.phone, ['phone', 'mobile']);
  trySet(['Email'], profile.email, ['email']);
  trySet(['Exp Location'], profile.expected_location, ['expected_location']);
  trySet(['Date of Birth'], profile.date_of_birth, ['dob', 'birth']);
  trySet(['First Degree'], profile.first_degree, ['degree']);
  trySet(['Annual Salary'], profile.annual_salary, ['salary']);
  trySet(['wechatsapp', 'whatsapp'], profile.whatsapp, ['whatsapp']);
  trySet(['Wechat'], profile.wechat, ['wechat']);
  trySet(['Anmelden'], profile.anmelden, ['anmelden']);
  trySet(['Marital status'], profile.marital_status, ['marital']);
  trySet(['Source'], profile.source, ['source']);
  trySet(['linkedin'], profile.linkedin, ['linkedin']);
  trySet(['Tags'], profile.tags, ['tags']);
  trySet(['Summary'], profile.summary, ['summary']);

  const firstWork = profile.work_experience?.[0] || {{}};
  trySet(['Company'], firstWork.company, ['company']);
  trySet(['Title'], firstWork.title, ['title']);
  trySet(['Work location'], firstWork.work_location, ['work_location']);
  trySet(['Responsibility'], firstWork.responsibility, ['responsibility']);
  if (firstWork.current_job) {{
    const current = Array.from(document.querySelectorAll('input[type=checkbox]'))
      .find(el => norm(el.parentElement?.textContent).includes('current job'));
    if (current && !current.checked) current.click();
  }}

  const firstEdu = profile.education?.[0] || {{}};
  trySet(['School'], firstEdu.school, ['school']);
  trySet(['Degree'], firstEdu.degree, ['degree']);
  trySet(['Major in'], firstEdu.major, ['major']);
  trySet(['Project Experience'], profile.project_experience, ['project']);
  trySet(['Creator'], profile.creator, ['creator']);

  console.log('CRM autofill complete. Labels not found:', misses);
}})();
"""


def write_review_html(review: dict[str, Any], path: Path) -> None:
    js = html.escape(build_autofill_js(review))
    body = html.escape(json.dumps(review, ensure_ascii=False, indent=2))
    path.write_text(
        f"""<!doctype html>
<meta charset="utf-8">
<title>Resume CRM Review</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; }}
pre {{ white-space: pre-wrap; background: #f6f6f6; padding: 16px; border: 1px solid #ddd; }}
textarea {{ width: 100%; height: 260px; }}
</style>
<h1>Resume CRM Review</h1>
<p>Review the JSON first. If it looks right, paste the autofill script into the CRM page console.</p>
<h2>Review JSON</h2>
<pre>{body}</pre>
<h2>Autofill Script</h2>
<textarea readonly>{js}</textarea>
""",
        encoding="utf-8",
    )


def run_self_check() -> None:
    sample = """Siti Aishah binti Mohamed
+65 9332 1819 | siti@example.com
Blk 123 Tampines Street 11 Singapore 521123
CMFAS M9A, GCE O Level, DBS Bank Pte Ltd
Date of Birth: 18 June 1988
"""
    review = extract_review(sample, model=DEFAULT_MODEL, timeout=1, no_model=True)
    profile = review["profile"]
    assert profile["phone"] == "+65 9332 1819"
    assert "Pte Ltd" in normalize_text(sample)
    assert "CMFAS M9A" in normalize_text(sample)
    assert "GCE O Level" in normalize_text(sample)
    assert profile["name"] == "Siti Aishah binti Mohamed"
    assert "industry" in review["missing_required"]
    assert profile["marital_status"] == ""
    assert profile["gender"] == ""

    old = coerce_profile(
        {
            "ground_truth": {
                "Name": "Tan Wei Ming",
                "Phone": "91234567",
                "Email": "tan@example.com",
                "Current Location": "Singapore",
                "Work Experience": [{"company": "OCBC Bank", "title": "Analyst", "from": "2020", "to": "Present"}],
            }
        }
    )
    scrubbed = coerce_profile(
        {"gender": "Not provided", "date_of_birth": "N/A", "annual_salary": "To be discussed with employer"}
    )
    assert scrubbed["gender"] == ""
    assert scrubbed["date_of_birth"] == ""
    assert scrubbed["annual_salary"] == ""

    old_review = build_review(old, "")
    assert old["work_experience"][0]["current_job"] is True
    assert "function" in old_review["missing_required"]
    assert "industry" in old_review["missing_required"]
    assert old["creator"] == ""
    print("self-check passed")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract resume data and prepare CRM profile autofill review.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--text", help="Raw pasted resume text.")
    src.add_argument("--text-file", type=Path, help="Path to a text file containing pasted resume text.")
    src.add_argument("--file", type=Path, help="Path to a .pdf, .docx, or .txt resume.")
    src.add_argument("--dir", type=Path, help="Batch: a merlion_resumes folder of candidate sub-folders.")
    parser.add_argument("--out-dir", type=Path, help="Batch output folder (default: <dir>/../crm_reviews).")
    parser.add_argument("--limit", type=int, default=0, help="Batch: only process the first N candidates.")
    parser.add_argument("--overwrite", action="store_true", help="Batch: re-process candidates that already have output.")
    parser.add_argument("--out", type=Path, help="Write review JSON here.")
    parser.add_argument("--review-html", type=Path, help="Write a human-review HTML file.")
    parser.add_argument("--autofill-js", type=Path, help="Write browser-console autofill JavaScript here.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--timeout", type=int, default=120, help="Ollama timeout seconds.")
    parser.add_argument("--no-model", action="store_true", help="Use deterministic extraction only.")
    parser.add_argument("--self-check", action="store_true", help="Run assert-based self-checks.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.self_check:
        run_self_check()
        return 0

    if args.dir:
        return run_batch(args)

    if args.text is not None:
        raw = args.text
    elif args.text_file:
        raw = args.text_file.read_text(encoding="utf-8", errors="replace")
    elif args.file:
        raw = read_text(args.file)
    else:
        raise SystemExit("Provide --text, --text-file, --file, or --self-check.")

    review = extract_review(raw, model=args.model, timeout=args.timeout, no_model=args.no_model)
    rendered = json.dumps(review, ensure_ascii=False, indent=2)
    print(rendered)

    if args.out:
        args.out.write_text(rendered + "\n", encoding="utf-8")
    if args.autofill_js:
        args.autofill_js.write_text(build_autofill_js(review), encoding="utf-8")
    if args.review_html:
        write_review_html(review, args.review_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
