"""
db_diagnostic.py

💅✨ FAIRY CODEMOTHER'S EXTRACTION DIAGNOSTIC TOOL ✨💅

This standalone script connects to your resume_extractions.db and helps you:

1. 🔍 FIND problems     — Missing data, wrong field placement, low-quality extractions
2. 🩺 DIAGNOSE causes   — Compare raw text vs extracted data to pinpoint failures
3. ✏️  CORRECT data      — Fix wrong extractions and log corrections for pattern learning
4. 📊 ANALYZE patterns  — Identify recurring extraction failures to improve code
5. 🔄 RE-EXTRACT        — Reprocess specific candidates after code improvements

Run this SEPARATELY from main.py — it's your investigation toolkit!

Usage:
    python db_diagnostic.py
"""

import os
import re
import json
import datetime
import sqlite3
from typing import Dict, List, Optional, Tuple
from db_manager import DatabaseManager
import config


# =============================================================================
# 🎨 DISPLAY HELPERS — Making diagnostics look GORGEOUS! 💄
# =============================================================================

def print_header(title: str, width: int = 70):
    """Print a fancy section header"""
    print("\n" + "╔" + "═" * width + "╗")
    print("║" + f"  {title}  ".center(width) + "║")
    print("╚" + "═" * width + "╝")


def print_divider(char: str = "─", width: int = 70):
    """Print a divider line"""
    print(char * width)


def print_field_comparison(field: str, extracted: str, raw_snippet: str = ""):
    """Print a field with its extracted value and optional raw text snippet"""
    status = "✅" if extracted else "❌ MISSING"
    # Truncate long values for display
    display_val = str(extracted)[:80] + "..." if extracted and len(str(extracted)) > 80 else extracted
    print(f"  {status} {field:20s}: {display_val or 'None'}")
    if raw_snippet:
        print(f"       📜 Raw text hint : {raw_snippet[:80]}...")


def colorize_match(text: str, keyword: str, context_chars: int = 60) -> str:
    """Find keyword in text and return surrounding context"""
    if not text or not keyword:
        return ""
    idx = text.lower().find(keyword.lower())
    if idx < 0:
        return ""
    start = max(0, idx - context_chars)
    end = min(len(text), idx + len(keyword) + context_chars)
    snippet = text[start:end]
    return f"...{snippet}..."


# =============================================================================
# 🔍 PROBLEM FINDER — The Detective Suite! 🕵️‍♀️
# =============================================================================

def find_all_problems(db: DatabaseManager) -> Dict:
    """
    🔍 Comprehensive scan for ALL types of extraction problems!
    
    Returns a structured report of every issue found.
    """
    report = {
        "missing_critical_fields": [],
        "wrong_field_suspects": [],
        "failed_extractions": [],
        "partial_extractions": [],
        "empty_sections": [],
        "raw_text_has_data_but_extraction_missed": [],
        "total_candidates": 0,
        "total_issues": 0
    }

    try:
        conn = db._connection

        # --- Count total candidates ---
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM structured_extractions")
        report["total_candidates"] = cursor.fetchone()["cnt"]

        # --- 1. Missing critical fields ---
        report["missing_critical_fields"] = db.get_missing_fields_report()

        # --- 2. Wrong field suspects ---
        report["wrong_field_suspects"] = db.get_wrong_field_suspects()

        # --- 3. Failed extractions ---
        cursor = conn.execute("""
            SELECT s.candidate_id, s.name, s.extraction_status, s.notes,
                   r.text_length, r.filenames
            FROM structured_extractions s
            LEFT JOIN raw_extractions r ON s.candidate_id = r.candidate_id
            WHERE s.extraction_status = 'Failed'
            ORDER BY s.candidate_id
        """)
        report["failed_extractions"] = [dict(row) for row in cursor.fetchall()]

        # --- 4. Partial extractions ---
        cursor = conn.execute("""
            SELECT s.candidate_id, s.name, s.extraction_status,
                   s.email, s.phone, s.date_of_birth,
                   CASE WHEN s.skills_raw IS NULL THEN 0 ELSE 1 END as has_skills,
                   CASE WHEN s.experience_raw IS NULL THEN 0 ELSE 1 END as has_exp,
                   CASE WHEN s.education_raw IS NULL THEN 0 ELSE 1 END as has_edu
            FROM structured_extractions s
            WHERE s.extraction_status = 'Partial'
            ORDER BY s.candidate_id
        """)
        report["partial_extractions"] = [dict(row) for row in cursor.fetchall()]

        # --- 5. Raw text has sections but extraction missed them ---
        # This is the GOLD MINE for finding extraction bugs! ⛏️
        section_keywords = {
            "skills_raw": ["SKILLS", "COMPETENCIES", "EXPERTISE", "PROFICIENCIES",
                          "TECHNICAL SKILLS", "KEY SKILLS", "CORE COMPETENCIES"],
            "experience_raw": ["EXPERIENCE", "EMPLOYMENT", "WORK HISTORY",
                              "CAREER HISTORY", "PROFESSIONAL EXPERIENCE",
                              "WORK EXPERIENCE"],
            "education_raw": ["EDUCATION", "QUALIFICATIONS", "ACADEMIC",
                             "UNIVERSITY", "DEGREE", "DIPLOMA"],
        }

        cursor = conn.execute("""
            SELECT s.candidate_id, s.name,
                   s.skills_raw, s.experience_raw, s.education_raw,
                   r.raw_text
            FROM structured_extractions s
            JOIN raw_extractions r ON s.candidate_id = r.candidate_id
            WHERE s.skills_raw IS NULL 
               OR s.experience_raw IS NULL 
               OR s.education_raw IS NULL
        """)

        for row in cursor.fetchall():
            raw_upper = (row["raw_text"] or "").upper()
            missed_sections = []

            for field, keywords in section_keywords.items():
                if not row[field]:  # Field is empty
                    found_keywords = [kw for kw in keywords if kw in raw_upper]
                    if found_keywords:
                        # Raw text HAS the section but extraction MISSED it!
                        missed_sections.append({
                            "field": field.replace("_raw", ""),
                            "keywords_found": found_keywords
                        })

            if missed_sections:
                report["raw_text_has_data_but_extraction_missed"].append({
                    "candidate_id": row["candidate_id"],
                    "name": row["name"] or "UNKNOWN",
                    "missed_sections": missed_sections
                })

        # Calculate total issues
        report["total_issues"] = (
            len(report["missing_critical_fields"]) +
            len(report["wrong_field_suspects"]) +
            len(report["failed_extractions"]) +
            len(report["raw_text_has_data_but_extraction_missed"])
        )

    except sqlite3.Error as e:
        print(f"❌ Database error during problem scan: {e}")

    return report


def print_problem_report(report: Dict):
    """
    📊 Print a beautiful, actionable problem report!
    """
    print_header("🔍 EXTRACTION PROBLEM REPORT")

    total = report["total_candidates"]
    issues = report["total_issues"]
    health = ((total - issues) / total * 100) if total > 0 else 0

    print(f"\n  📊 Total candidates in DB: {total}")
    print(f"  🚨 Total issues found:     {issues}")
    print(f"  💚 Health score:            {health:.1f}%")

    # --- Failed Extractions ---
    failed = report["failed_extractions"]
    if failed:
        print(f"\n  🔴 FAILED EXTRACTIONS ({len(failed)}):")
        print_divider("─", 60)
        for f in failed[:15]:
            raw_len = f.get("text_length") or 0
            note = f.get("notes", "")[:50]
            print(f"    ID {f['candidate_id']:>6} | "
                  f"Raw: {raw_len:>5} chars | "
                  f"{note}")
        if len(failed) > 15:
            print(f"    ... and {len(failed) - 15} more")

    # --- Missing Critical Fields ---
    missing = report["missing_critical_fields"]
    if missing:
        print(f"\n  🟡 MISSING CRITICAL FIELDS ({len(missing)}):")
        print_divider("─", 60)

        # Tally which fields are most commonly missing
        field_tally = {}
        for m in missing:
            for field in m["missing_fields"]:
                field_tally[field] = field_tally.get(field, 0) + 1

        print("    Most commonly missing:")
        for field, count in sorted(field_tally.items(), key=lambda x: -x[1]):
            bar_len = min(20, int(count / max(1, total) * 100))
            bar = "█" * bar_len
            print(f"      {field:20s} {bar} {count} ({count/total*100:.0f}%)")

    # --- Wrong Field Suspects ---
    suspects = report["wrong_field_suspects"]
    if suspects:
        print(f"\n  🕵️ WRONG FIELD SUSPECTS ({len(suspects)}):")
        print_divider("─", 60)
        for s in suspects[:10]:
            print(f"    ID {s['candidate_id']:>6}:")
            for issue in s["issues"]:
                print(f"      ⚠️ {issue}")
        if len(suspects) > 10:
            print(f"    ... and {len(suspects) - 10} more")

    # --- Extraction Missed Sections ---
    missed = report["raw_text_has_data_but_extraction_missed"]
    if missed:
        print(f"\n  ⛏️ RAW TEXT HAS DATA BUT EXTRACTION MISSED IT ({len(missed)}):")
        print_divider("─", 60)
        print("    These candidates have section headers in raw text but")
        print("    the extraction returned NULL — likely a regex/AI failure!")
        print()

        # Tally which sections are most commonly missed
        section_tally = {}
        for m in missed:
            for sec in m["missed_sections"]:
                section_tally[sec["field"]] = section_tally.get(sec["field"], 0) + 1

        print("    Most commonly missed sections:")
        for section, count in sorted(section_tally.items(), key=lambda x: -x[1]):
            print(f"      {section:20s}: {count} candidates affected")

        print("\n    Sample candidates:")
        for m in missed[:5]:
            sections = ", ".join([s["field"] for s in m["missed_sections"]])
            print(f"      ID {m['candidate_id']:>6} ({m['name']}): missed [{sections}]")

    # --- Health Summary ---
    print_header("💊 RECOMMENDED ACTIONS")
    actions = []

    if missed:
        actions.append(
            "🔧 FIX SECTION DETECTION: Your regex patterns are missing "
            f"{len(missed)} resume sections that clearly exist in the raw text. "
            "Run option [2] to inspect specific candidates."
        )

    if suspects:
        actions.append(
            "🔍 CHECK FIELD BOUNDARIES: Data is landing in wrong fields for "
            f"{len(suspects)} candidates. This often means section headers "
            "aren't being detected properly."
        )

    if failed:
        # Check if failures have raw text
        no_text = sum(1 for f in failed if (f.get("text_length") or 0) < 50)
        has_text = len(failed) - no_text
        if no_text > 0:
            actions.append(
                f"📄 FIX PDF EXTRACTION: {no_text} candidates failed because "
                "the PDF text extraction itself returned too little text. "
                "Check OCR settings or PDF extraction method."
            )
        if has_text > 0:
            actions.append(
                f"🤖 IMPROVE PARSING: {has_text} candidates have raw text but "
                "extraction still failed. The text is there — the parser needs "
                "improvement."
            )

    field_tally = {}
    for m in missing:
        for field in m["missing_fields"]:
            field_tally[field] = field_tally.get(field, 0) + 1

    worst_field = max(field_tally.items(), key=lambda x: x[1]) if field_tally else None
    if worst_field and worst_field[1] > 3:
        actions.append(
            f"🎯 PRIORITY FIX: '{worst_field[0]}' is missing for {worst_field[1]} "
            f"candidates — focus regex/AI improvements on this field first."
        )

    if not actions:
        actions.append("🎉 Looking great! No major issues detected.")

    for i, action in enumerate(actions, 1):
        print(f"\n  {i}. {action}")

    print()


# =============================================================================
# 🩺 DEEP INVESTIGATION — Compare raw text vs extraction! 🔬
# =============================================================================

def investigate_candidate(db: DatabaseManager, candidate_id: int):
    """
    🩺 Deep-dive investigation into a single candidate's extraction!
    
    Shows raw text side-by-side with extracted data so you can see
    exactly what went wrong and where.
    """
    print_header(f"🩺 INVESTIGATING CANDIDATE {candidate_id}")

    # Get structured data
    try:
        cursor = db._connection.execute("""
            SELECT * FROM structured_extractions
            WHERE candidate_id = ?
            ORDER BY created_at DESC LIMIT 1
        """, (candidate_id,))
        structured = cursor.fetchone()
    except sqlite3.Error:
        structured = None

    if not structured:
        print(f"  ❌ No structured extraction found for candidate {candidate_id}")
        return

    structured = dict(structured)

    # Get raw text
    raw_text = db.get_raw_text_for_candidate(candidate_id)
    if not raw_text:
        print(f"  ❌ No raw text found for candidate {candidate_id}")
        return

    # --- Show extraction overview ---
    print(f"\n  📊 Status:      {structured.get('extraction_status', 'Unknown')}")
    print(f"  🤖 AI Assisted: {'Yes' if structured.get('ai_assisted') else 'No'}")
    print(f"  📝 Notes:       {structured.get('notes', 'None')[:80]}")
    print(f"  📏 Raw text:    {len(raw_text)} characters")

    # --- Field-by-field comparison ---
    print(f"\n  {'─' * 65}")
    print(f"  📋 FIELD-BY-FIELD ANALYSIS:")
    print(f"  {'─' * 65}")

    fields_to_check = {
        "name": {
            "db_col": "name",
            "search_hint": ["Name", "NAME"],
            "context_label": "Header area (first 500 chars)"
        },
        "email": {
            "db_col": "email",
            "search_hint": ["@", "Email", "E-mail"],
            "regex_check": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        },
        "phone": {
            "db_col": "phone",
            "search_hint": ["Phone", "Mobile", "Tel", "Contact", "HP"],
            "regex_check": r'[\+]?[\d\s\-\(\)]{8,15}'
        },
        "date_of_birth": {
            "db_col": "date_of_birth",
            "search_hint": ["DOB", "Date of Birth", "Birthday", "Born", "D.O.B"],
            "regex_check": r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}'
        },
        "skills": {
            "db_col": "skills_raw",
            "search_hint": ["Skills", "Competencies", "Expertise",
                           "Technical Skills", "Key Skills", "Proficiencies"]
        },
        "experience": {
            "db_col": "experience_raw",
            "search_hint": ["Experience", "Employment", "Work History",
                           "Career", "Professional Experience"]
        },
        "education": {
            "db_col": "education_raw",
            "search_hint": ["Education", "Qualifications", "Academic",
                           "University", "Degree", "Diploma"]
        },
    }

    raw_upper = raw_text.upper()

    for field_name, info in fields_to_check.items():
        extracted_value = structured.get(info["db_col"])

        # Check if keywords exist in raw text
        found_keywords = [kw for kw in info.get("search_hint", [])
                         if kw.upper() in raw_upper]

        # Try regex check if available
        regex_match = None
        if info.get("regex_check"):
            match = re.search(info["regex_check"], raw_text)
            if match:
                regex_match = match.group(0).strip()

        # Show the comparison
        status = "✅" if extracted_value else "❌"
        display_val = str(extracted_value)[:70] if extracted_value else "MISSING"
        print(f"\n  {status} {field_name.upper()}")
        print(f"      Extracted:  {display_val}")

        if not extracted_value:
            # It's missing — show what we found in raw text
            if found_keywords:
                print(f"      ⚠️ Keywords FOUND in raw: {found_keywords}")
                # Show context around the keyword
                for kw in found_keywords[:2]:
                    snippet = colorize_match(raw_text, kw, 80)
                    if snippet:
                        print(f"      📜 Context:  {snippet}")
                print(f"      💡 DIAGNOSIS: Section exists but extraction FAILED!")
            else:
                print(f"      ℹ️ No section keywords found in raw text")
                print(f"      💡 DIAGNOSIS: Resume may not contain this info")

            if regex_match:
                print(f"      🔍 Regex found: '{regex_match}'")
                print(f"      💡 Regex CAN find it but pipeline didn't capture it!")

        elif found_keywords and regex_match and regex_match not in str(extracted_value):
            # Extracted something, but there might be MORE in the text
            print(f"      🔍 Regex also found: '{regex_match[:60]}'")
            if regex_match != extracted_value:
                print(f"      ⚠️ Different value found — possible incorrect extraction!")

    # --- Show raw text header (usually contains contact info) ---
    print(f"\n  {'─' * 65}")
    print(f"  📜 RAW TEXT — FIRST 800 CHARACTERS (contact info area):")
    print(f"  {'─' * 65}")
    header_text = raw_text[:800]
    for line in header_text.split('\n')[:25]:
        stripped = line.strip()
        if stripped:
            print(f"    | {stripped[:75]}")

    print(f"\n  {'─' * 65}")
    print(f"  📜 RAW TEXT — SECTION HEADERS DETECTED:")
    print(f"  {'─' * 65}")

    # Find all section-like headers in the raw text
    header_pattern = r'^[\s]*([A-Z][A-Z\s&/]{3,40})\s*[:\n]?\s*$'
    headers_found = re.findall(header_pattern, raw_text, re.MULTILINE)
    unique_headers = list(dict.fromkeys([h.strip() for h in headers_found if len(h.strip()) > 3]))

    if unique_headers:
        for h in unique_headers[:20]:
            print(f"    📌 {h}")
    else:
        print(f"    ⚠️ No clear section headers detected!")
        print(f"    💡 This resume may use non-standard formatting")


# =============================================================================
# ✏️ CORRECTION SYSTEM — Fix and learn! 📝
# =============================================================================

def correct_candidate(db: DatabaseManager, candidate_id: int):
    """
    ✏️ Interactively correct extraction fields for a candidate!
    
    Corrections are logged in the extraction_log table so you can
    track what was wrong and learn from patterns.
    """
    # First, show the investigation so user knows what to fix
    investigate_candidate(db, candidate_id)

    print_header(f"✏️ CORRECT CANDIDATE {candidate_id}")

    correctable_fields = [
        ("name", "Name"),
        ("email", "Email"),
        ("phone", "Phone"),
        ("date_of_birth", "Date of Birth"),
        ("location", "Location"),
        ("skills", "Skills (pipe-separated: Python | Java | SQL)"),
    ]

    corrections_made = 0

    for field_key, field_label in correctable_fields:
        # Get current value
        try:
            col = {"skills": "skills_raw", "date_of_birth": "date_of_birth"}.get(
                field_key, field_key
            )
            cursor = db._connection.execute(f"""
                SELECT {col} FROM structured_extractions
                WHERE candidate_id = ?
                ORDER BY created_at DESC LIMIT 1
            """, (candidate_id,))
            row = cursor.fetchone()
            current = row[col] if row else None
        except sqlite3.Error:
            current = None

        display_current = str(current)[:60] if current else "[EMPTY]"
        print(f"\n  {field_label}: {display_current}")
        new_value = input("  New value (Enter to skip, 'null' to clear): ").strip()

        if new_value == "":
            continue  # Skip, keep current value
        elif new_value.lower() == "null":
            new_value = None

        # Apply the correction
        if new_value != current:
            db.update_field(candidate_id, field_key, new_value or "")
            corrections_made += 1
            print(f"  ✅ Updated!")

    if corrections_made > 0:
        # Mark as reviewed
        db.mark_as_reviewed(candidate_id, f"Manually corrected {corrections_made} fields")
        print(f"\n  🎉 {corrections_made} corrections saved and logged!")
        print(f"  📝 Candidate marked as reviewed in the database.")
    else:
        print(f"\n  ℹ️ No corrections made.")


# =============================================================================
# 📊 PATTERN ANALYSIS — Find recurring problems! 🔬
# =============================================================================

def analyze_failure_patterns(db: DatabaseManager):
    """
    📊 Analyze patterns in extraction failures to identify systemic issues!
    
    This tells you WHERE to focus your code improvements for maximum impact.
    """
    print_header("📊 FAILURE PATTERN ANALYSIS")

    try:
        conn = db._connection

        # --- 1. Field-level success rates ---
        print("\n  🎯 FIELD EXTRACTION SUCCESS RATES:")
        print_divider("─", 60)

        fields = [
            ("name", "Name"),
            ("email", "Email"),
            ("phone", "Phone"),
            ("date_of_birth", "Date of Birth"),
            ("skills_raw", "Skills"),
            ("experience_raw", "Experience"),
            ("education_raw", "Education"),
            ("location", "Location"),
        ]

        cursor = conn.execute("SELECT COUNT(*) as cnt FROM structured_extractions")
        total = cursor.fetchone()["cnt"]

        field_stats = []
        for col, label in fields:
            cursor = conn.execute(f"""
                SELECT COUNT(*) as cnt FROM structured_extractions
                WHERE {col} IS NOT NULL AND {col} != ''
            """)
            found = cursor.fetchone()["cnt"]
            rate = (found / total * 100) if total > 0 else 0
            field_stats.append((label, found, total, rate))

            bar_len = int(rate / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            color = "✅" if rate >= 80 else "🟡" if rate >= 50 else "🔴"
            print(f"    {color} {label:15s} {bar} {rate:5.1f}%  ({found}/{total})")

        # --- 2. AI vs Regex effectiveness ---
        print(f"\n\n  🤖 AI vs REGEX EFFECTIVENESS:")
        print_divider("─", 60)

        cursor = conn.execute("""
            SELECT 
                ai_assisted,
                extraction_status,
                COUNT(*) as cnt
            FROM structured_extractions
            GROUP BY ai_assisted, extraction_status
            ORDER BY ai_assisted, extraction_status
        """)

        ai_stats = {"AI": {}, "Regex": {}}
        for row in cursor.fetchall():
            method = "AI" if row["ai_assisted"] else "Regex"
            ai_stats[method][row["extraction_status"]] = row["cnt"]

        for method, statuses in ai_stats.items():
            method_total = sum(statuses.values())
            complete = statuses.get("Complete", 0)
            success = statuses.get("Success", 0)
            good = complete + success
            rate = (good / method_total * 100) if method_total > 0 else 0
            print(f"    {'🤖' if method == 'AI' else '⚡'} {method:8s}: "
                  f"{good}/{method_total} successful ({rate:.1f}%)")
            for status, count in sorted(statuses.items()):
                print(f"        {status}: {count}")

        # --- 3. Text length vs success correlation ---
        print(f"\n\n  📏 TEXT LENGTH vs EXTRACTION SUCCESS:")
        print_divider("─", 60)

        cursor = conn.execute("""
            SELECT 
                CASE 
                    WHEN r.text_length < 200 THEN 'Very Short (<200)'
                    WHEN r.text_length < 500 THEN 'Short (200-500)'
                    WHEN r.text_length < 2000 THEN 'Medium (500-2K)'
                    WHEN r.text_length < 5000 THEN 'Long (2K-5K)'
                    ELSE 'Very Long (5K+)'
                END as length_bucket,
                s.extraction_status,
                COUNT(*) as cnt
            FROM structured_extractions s
            JOIN raw_extractions r ON s.candidate_id = r.candidate_id
            GROUP BY length_bucket, s.extraction_status
            ORDER BY r.text_length
        """)

        length_data = {}
        for row in cursor.fetchall():
            bucket = row["length_bucket"]
            if bucket not in length_data:
                length_data[bucket] = {}
            length_data[bucket][row["extraction_status"]] = row["cnt"]

        for bucket, statuses in length_data.items():
            bucket_total = sum(statuses.values())
            failed = statuses.get("Failed", 0)
            fail_rate = (failed / bucket_total * 100) if bucket_total > 0 else 0
            indicator = "🔴" if fail_rate > 30 else "🟡" if fail_rate > 10 else "🟢"
            print(f"    {indicator} {bucket:25s}: {bucket_total:3d} total, "
                  f"{failed:2d} failed ({fail_rate:.0f}%)")

        # --- 4. Extraction log analysis ---
        print(f"\n\n  🔎 EXTRACTION METHOD SUCCESS BY FIELD:")
        print_divider("─", 60)

        cursor = conn.execute("""
            SELECT 
                field_name,
                extraction_method,
                COUNT(*) as total,
                SUM(was_successful) as successes
            FROM extraction_log
            GROUP BY field_name, extraction_method
            ORDER BY field_name, extraction_method
        """)

        current_field = None
        for row in cursor.fetchall():
            if row["field_name"] != current_field:
                current_field = row["field_name"]
                print(f"\n    📌 {current_field.upper()}:")

            total_attempts = row["total"]
            successes = row["successes"] or 0
            rate = (successes / total_attempts * 100) if total_attempts > 0 else 0
            indicator = "✅" if rate >= 80 else "🟡" if rate >= 50 else "🔴"
            print(f"       {indicator} {row['extraction_method']:20s}: "
                  f"{successes}/{total_attempts} ({rate:.0f}%)")

        # --- 5. Corrections analysis ---
        cursor = conn.execute("""
            SELECT COUNT(*) as cnt FROM extraction_log
            WHERE extraction_method = 'manual_correction'
        """)
        corrections_count = cursor.fetchone()["cnt"]

        if corrections_count > 0:
            print(f"\n\n  ✏️ MANUAL CORRECTIONS MADE: {corrections_count}")
            print_divider("─", 60)

            cursor = conn.execute("""
                SELECT field_name, COUNT(*) as cnt
                FROM extraction_log
                WHERE extraction_method = 'manual_correction'
                GROUP BY field_name
                ORDER BY cnt DESC
            """)
            for row in cursor.fetchall():
                print(f"    {row['field_name']:20s}: {row['cnt']} corrections")

            print(f"\n    💡 Fields with many corrections = where your extraction ")
            print(f"       patterns need the most improvement!")

    except sqlite3.Error as e:
        print(f"  ❌ Analysis failed: {e}")


# =============================================================================
# 📋 GENERATE IMPROVEMENT REPORT — Actionable code fixes! 🔧
# =============================================================================

def generate_improvement_report(db: DatabaseManager):
    """
    📋 Generate a SPECIFIC, actionable report on what code to improve!
    
    This connects database findings to ACTUAL code locations.
    """
    print_header("📋 CODE IMPROVEMENT RECOMMENDATIONS")

    report = find_all_problems(db)
    missed = report["raw_text_has_data_but_extraction_missed"]
    suspects = report["wrong_field_suspects"]
    failed = report["failed_extractions"]

    rec_num = 1

    # --- Skills extraction issues ---
    skills_missed = sum(1 for m in missed
                       if any(s["field"] == "skills" for s in m["missed_sections"]))
    if skills_missed > 0:
        print(f"\n  {rec_num}. 🔧 SKILLS EXTRACTION ({skills_missed} candidates affected)")
        print(f"     File: main.py → _format_skills_for_export()")
        print(f"     File: main.py → _extract_section_raw()")
        print(f"     File: ai_extractor.py → extract_deep_fields()")
        print(f"     Problem: Section header regex isn't matching these resumes.")
        print(f"     Action: Run option [2] on affected candidates to see the")
        print(f"             actual section headers in their raw text, then add")
        print(f"             those header variations to the keyword lists.")
        print(f"     Quick check IDs: {[m['candidate_id'] for m in missed if any(s['field'] == 'skills' for s in m['missed_sections'])][:5]}")
        rec_num += 1

    # --- Experience extraction issues ---
    exp_missed = sum(1 for m in missed
                    if any(s["field"] == "experience" for s in m["missed_sections"]))
    if exp_missed > 0:
        print(f"\n  {rec_num}. 🔧 EXPERIENCE EXTRACTION ({exp_missed} candidates affected)")
        print(f"     File: main.py → _format_experience_for_export()")
        print(f"     File: ai_extractor.py → _fix_malformed_lists()")
        print(f"     Problem: Work experience sections exist but aren't captured.")
        print(f"     Action: Check if the section headers use non-standard names")
        print(f"             (e.g., 'Career Summary', 'Job History', 'Positions Held').")
        print(f"     Quick check IDs: {[m['candidate_id'] for m in missed if any(s['field'] == 'experience' for s in m['missed_sections'])][:5]}")
        rec_num += 1

    # --- Education extraction issues ---
    edu_missed = sum(1 for m in missed
                    if any(s["field"] == "education" for s in m["missed_sections"]))
    if edu_missed > 0:
        print(f"\n  {rec_num}. 🔧 EDUCATION EXTRACTION ({edu_missed} candidates affected)")
        print(f"     File: main.py → _format_education_for_export()")
        print(f"     Problem: Education sections exist but aren't captured.")
        print(f"     Action: Check for non-standard headers like 'Academic Background',")
        print(f"             'Scholastic Record', 'Training & Education'.")
        print(f"     Quick check IDs: {[m['candidate_id'] for m in missed if any(s['field'] == 'education' for s in m['missed_sections'])][:5]}")
        rec_num += 1

    # --- Wrong field placement ---
    if suspects:
        print(f"\n  {rec_num}. 🔧 FIELD BOUNDARY DETECTION ({len(suspects)} candidates)")
        print(f"     File: main.py → _extract_with_mega_regex()")
        print(f"     File: main.py → _is_valid_extracted_name()")
        print(f"     File: ai_extractor.py → extract_header_fields()")
        print(f"     Problem: Data is being placed in the wrong field.")
        print(f"     Common causes:")
        print(f"       - Email in name field → name regex too greedy")
        print(f"       - Name too long → section header captured as name")
        print(f"       - Phone in email → field parsing order wrong")
        print(f"     Quick check IDs: {[s['candidate_id'] for s in suspects[:5]]}")
        rec_num += 1

    # --- PDF extraction failures ---
    no_text_failures = [f for f in failed if (f.get("text_length") or 0) < 100]
    if no_text_failures:
        print(f"\n  {rec_num}. 🔧 PDF TEXT EXTRACTION ({len(no_text_failures)} candidates)")
        print(f"     File: document_parser.py")
        print(f"     File: config.py → OCR settings")
        print(f"     Problem: PDF/DOCX text extraction returned very little text.")
        print(f"     Possible causes:")
        print(f"       - Scanned/image PDF without OCR")
        print(f"       - Password-protected PDF")
        print(f"       - Non-standard PDF encoding")
        print(f"     Action: Check OCR_ALWAYS_RUN and HYBRID_EXTRACTION settings.")
        print(f"     Quick check IDs: {[f['candidate_id'] for f in no_text_failures[:5]]}")
        rec_num += 1

    if rec_num == 1:
        print(f"\n  🎉 No major code improvements needed! Your extraction is fabulous!")

    # --- Save report to file ---
    report_path = f"improvement_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Extraction Improvement Report\n")
            f.write(f"Generated: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Total candidates: {report['total_candidates']}\n")
            f.write(f"Total issues: {report['total_issues']}\n\n")

            f.write(f"Failed extractions: {len(failed)}\n")
            for item in failed:
                f.write(f"  ID {item['candidate_id']}: {item.get('notes', '')}\n")

            f.write(f"\nMissed sections (regex failure): {len(missed)}\n")
            for item in missed:
                sections = ", ".join([s["field"] for s in item["missed_sections"]])
                f.write(f"  ID {item['candidate_id']} ({item['name']}): {sections}\n")

            f.write(f"\nWrong field suspects: {len(suspects)}\n")
            for item in suspects:
                issues = "; ".join(item["issues"])
                f.write(f"  ID {item['candidate_id']}: {issues}\n")

        print(f"\n  📄 Full report saved to: {report_path}")
    except IOError as e:
        print(f"\n  ⚠️ Could not save report file: {e}")


# =============================================================================
# 🔄 BATCH OPERATIONS — Efficient bulk tasks! ⚡
# =============================================================================

def list_candidates_for_review(db: DatabaseManager, filter_type: str = "all"):
    """
    📋 List candidates that need attention, with smart filtering.
    """
    print_header("📋 CANDIDATES NEEDING REVIEW")

    try:
        conn = db._connection

        if filter_type == "failed":
            where = "WHERE s.extraction_status = 'Failed'"
            title = "FAILED EXTRACTIONS"
        elif filter_type == "partial":
            where = "WHERE s.extraction_status = 'Partial'"
            title = "PARTIAL EXTRACTIONS"
        elif filter_type == "unreviewed":
            where = "WHERE s.reviewed = 0"
            title = "UNREVIEWED EXTRACTIONS"
        elif filter_type == "missing_contact":
            where = "WHERE (s.email IS NULL OR s.phone IS NULL)"
            title = "MISSING CONTACT INFO"
        else:
            where = ""
            title = "ALL CANDIDATES"

        cursor = conn.execute(f"""
            SELECT s.candidate_id, s.name, s.email, s.phone,
                   s.extraction_status, s.ai_assisted, s.reviewed,
                   r.text_length
            FROM structured_extractions s
            LEFT JOIN raw_extractions r ON s.candidate_id = r.candidate_id
            {where}
            ORDER BY s.candidate_id
        """)

        results = cursor.fetchall()

        print(f"\n  📊 {title}: {len(results)} candidates\n")

        if not results:
            print("  🎉 Nothing to show — all clear!")
            return []

        # Table header
        print(f"  {'ID':>6} | {'Name':25s} | {'Status':10s} | {'AI':3s} | {'Rev':3s} | {'Raw':>6s}")
        print(f"  {'─'*6}-+-{'─'*25}-+-{'─'*10}-+-{'─'*3}-+-{'─'*3}-+-{'─'*6}")

        candidate_ids = []
        for row in results:
            cid = row["candidate_id"]
            candidate_ids.append(cid)
            name = (row["name"] or "UNKNOWN")[:25]
            status = row["extraction_status"] or "?"
            ai = "🤖" if row["ai_assisted"] else "  "
            rev = "✅" if row["reviewed"] else "  "
            raw_len = row["text_length"] or 0

            status_icon = {
                "Complete": "🟢", "Success": "🟢",
                "Partial": "🟡", "Failed": "🔴"
            }.get(status, "⚪")

            print(f"  {cid:>6} | {name:25s} | {status_icon}{status:8s} | {ai:3s} | {rev:3s} | {raw_len:>6}")

        return candidate_ids

    except sqlite3.Error as e:
        print(f"  ❌ Query failed: {e}")
        return []


# =============================================================================
# 🎭 MAIN MENU — The Show! 🌟
# =============================================================================

def main():
    """
    🎭 The Main Diagnostic Menu!
    """
    db_path = config.DATABASE_FILE if hasattr(config, 'DATABASE_FILE') else "resume_extractions.db"

    if not os.path.exists(db_path):
        print(f"\n  ❌ Database not found: {db_path}")
        print(f"  💡 Run your main extraction pipeline first to populate the database!")
        return

    db = DatabaseManager(db_path)

    while True:
        print_header("🔮 FAIRY CODEMOTHER'S DIAGNOSTIC TOOLKIT 🔮")
        print("""
  🔍 FIND PROBLEMS:
     [1] 📊 Full Problem Scan — find ALL extraction issues
     [2] 🩺 Investigate Candidate — deep-dive into one candidate
     [3] 📋 List Candidates — browse by status/filter

  🔧 FIX & IMPROVE:
     [4] ✏️  Correct Candidate — fix wrong extractions
     [5] 📊 Failure Pattern Analysis — find recurring problems
     [6] 📋 Code Improvement Report — what to fix and where

  📈 DATABASE:
     [7] 📊 Database Stats — overview of all data
     [8] 🔍 Search Raw Text — find keywords across all resumes
     [9] 📤 Export to JSON — backup database to JSON file

     [0] 👋 Exit
        """)

        choice = input("  💅 What's it gonna be? (0-9): ").strip()

        if choice == "1":
            report = find_all_problems(db)
            print_problem_report(report)

        elif choice == "2":
            try:
                cid = int(input("\n  Enter candidate ID: ").strip())
                investigate_candidate(db, cid)
            except ValueError:
                print("  ❌ Please enter a valid number, darling!")

        elif choice == "3":
            print("\n  Filter by:")
            print("    [a] All candidates")
            print("    [f] Failed only")
            print("    [p] Partial only")
            print("    [u] Unreviewed only")
            print("    [c] Missing contact info")
            f = input("  Your choice: ").strip().lower()
            filter_map = {"a": "all", "f": "failed", "p": "partial",
                         "u": "unreviewed", "c": "missing_contact"}
            list_candidates_for_review(db, filter_map.get(f, "all"))

        elif choice == "4":
            try:
                cid = int(input("\n  Enter candidate ID to correct: ").strip())
                correct_candidate(db, cid)
            except ValueError:
                print("  ❌ Please enter a valid number, darling!")

        elif choice == "5":
            analyze_failure_patterns(db)

        elif choice == "6":
            generate_improvement_report(db)

        elif choice == "7":
            db.print_stats_report()

        elif choice == "8":
            keyword = input("\n  Search keyword: ").strip()
            if keyword:
                results = db.search_raw_text(keyword, limit=15)
                print(f"\n  🔍 Found {len(results)} candidates with '{keyword}':\n")
                for r in results:
                    print(f"    ID {r['candidate_id']:>6} ({r['name']}): "
                          f"{r.get('status', '?')}")
                    if r.get("snippet"):
                        print(f"             {r['snippet'][:80]}")
            else:
                print("  ❌ Please enter a keyword, honey!")

        elif choice == "9":
            output = db.export_to_json()
            if output:
                print(f"\n  📤 Exported to: {output}")

        elif choice == "0":
            db.close()
            print("\n  👋 Stay fabulous, darling! ✨\n")
            break

        else:
            print("  ❌ Invalid choice, try again sweetie!")

        input("\n  Press Enter to continue...")


if __name__ == "__main__":
    main()