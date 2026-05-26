"""
db_manager.py

💅✨ FAIRY CODEMOTHER'S DATABASE MANAGER DELUXE! ✨💅

This module manages a local SQLite database for storing BOTH raw and structured
resume extraction data. Think of it as the backstage vault where we keep ALL
the receipts — the original raw text AND the glammed-up structured output! 🎭

Three-layer architecture:
  1. raw_extractions     — The "before" photo: raw PDF/DOCX text as-is
  2. structured_extractions — The "after glam": parsed, structured fields
  3. extraction_log      — The detective's notebook: what method extracted what

This module integrates with the existing AiMerlion pipeline WITHOUT replacing
the current CSV/JSON export. It's an ADDITION, not a replacement, darling! 🌟

Usage:
    from db_manager import DatabaseManager
    db = DatabaseManager()           # Uses default 'resume_extractions.db'
    db.save_extraction(result, raw_text, folder_path)
    db.get_extraction_diagnostics(candidate_id=12345)
    db.close()
"""

import sqlite3
import json
import os
import datetime
import logging
import coloredlogs
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager

# Setup logging with our signature style 💄
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger,
                    fmt='%(asctime)s - 🗄️ %(levelname)s - %(message)s')


class DatabaseManager:
    """
    🏛️ THE DATABASE DIVA!
    
    She stores everything — raw text, structured data, and extraction logs —
    in a single SQLite file. No server setup, no configuration drama.
    Just pure, elegant data storage! 💎
    
    Thread Safety Note:
        SQLite connections are NOT thread-safe by default.
        This manager uses check_same_thread=False for flexibility,
        but if you're doing multi-threaded processing, use the
        get_connection() context manager for each thread.
    """

    # ==========================================================================
    # 🏗️ SCHEMA VERSION — Increment this when you change the schema!
    # Like a fashion season number — Spring 2026, darling! 💃
    # ==========================================================================
    SCHEMA_VERSION = 1

    def __init__(self, db_path: str = "resume_extractions.db"):
        """
        🎀 Initialize the Database Manager!
        
        Args:
            db_path: Path to the SQLite database file.
                     Defaults to 'resume_extractions.db' in the current directory.
                     The file will be CREATED if it doesn't exist — no drama needed!
        """
        self.db_path = db_path
        self._connection = None
        
        # 🎬 ACT 1: Open the connection and set up the stage!
        try:
            self._connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,    # Allows usage across threads
                timeout=30                   # Wait up to 30s if DB is locked
            )
            # Enable WAL mode for better concurrent read performance
            # Think of WAL as a VIP lane — readers don't block writers! 🚗💨
            self._connection.execute("PRAGMA journal_mode=WAL")
            # Enable foreign key enforcement (SQLite has it OFF by default!)
            self._connection.execute("PRAGMA foreign_keys=ON")
            # Row factory for dict-like access to query results
            self._connection.row_factory = sqlite3.Row
            
            logger.info(f"✨ Database connected: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise

        # 🎬 ACT 2: Create tables if they don't exist (first-run setup)
        self._initialize_schema()

    # ==========================================================================
    # 🔌 CONNECTION MANAGEMENT
    # ==========================================================================

    @contextmanager
    def get_connection(self):
        """
        🎭 Context manager for safe database operations.
        
        Usage:
            with db.get_connection() as conn:
                conn.execute("INSERT INTO ...")
                
        Auto-commits on success, auto-rollbacks on failure.
        Like a safety net under a trapeze artist! 🎪
        """
        conn = self._connection
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"❌ Database transaction failed, rolled back: {e}")
            raise

    def close(self):
        """
        👋 Gracefully close the database connection.
        Always call this when you're done, or use 'with' statements!
        """
        if self._connection:
            try:
                self._connection.close()
                logger.info("👋 Database connection closed gracefully.")
            except sqlite3.Error as e:
                logger.error(f"⚠️ Error closing database: {e}")
            finally:
                self._connection = None

    # ==========================================================================
    # 🏗️ SCHEMA INITIALIZATION
    # ==========================================================================

    def _initialize_schema(self):
        """
        🏗️ Create all tables and indexes if they don't exist!
        
        This is idempotent (safe to call multiple times) thanks to
        IF NOT EXISTS. Like putting on foundation — you can reapply
        without ruining the look! 💄
        """
        try:
            with self.get_connection() as conn:
                # ----------------------------------------------------------
                # 📸 TABLE 1: raw_extractions
                # The "before" photo — stores the ORIGINAL text from PDFs
                # ----------------------------------------------------------
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS raw_extractions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        candidate_id INTEGER,
                        folder_path TEXT NOT NULL,
                        filenames TEXT NOT NULL,
                        raw_text TEXT,
                        resume_language TEXT DEFAULT 'English',
                        text_length INTEGER DEFAULT 0,
                        pdf_page_count INTEGER,
                        extraction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(candidate_id, filenames)
                    )
                """)

                # ----------------------------------------------------------
                # 💃 TABLE 2: structured_extractions
                # The "after glam" — parsed, structured, validated fields
                # ----------------------------------------------------------
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS structured_extractions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        candidate_id INTEGER NOT NULL,
                        raw_extraction_id INTEGER,

                        -- 👤 Personal Info
                        name TEXT,
                        email TEXT,
                        phone TEXT,
                        date_of_birth TEXT,
                        location TEXT,
                        nationality TEXT,

                        -- 💼 Professional Content (raw = original string, json = parsed)
                        skills_raw TEXT,
                        skills_json TEXT,
                        experience_raw TEXT,
                        experience_json TEXT,
                        education_raw TEXT,
                        education_json TEXT,

                        -- 📋 Additional Fields
                        summary TEXT,
                        certifications TEXT,
                        languages TEXT,
                        projects TEXT,
                        achievements TEXT,
                        references_info TEXT,
                        hobbies TEXT,

                        -- 🔍 META: The detective work!
                        extraction_method TEXT DEFAULT 'Unknown',
                        ai_assisted INTEGER DEFAULT 0,
                        extraction_status TEXT DEFAULT 'Pending',
                        notes TEXT DEFAULT '',

                        -- 📅 Audit trail
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        reviewed INTEGER DEFAULT 0,
                        review_notes TEXT,

                        FOREIGN KEY (raw_extraction_id)
                            REFERENCES raw_extractions(id)
                            ON DELETE SET NULL
                    )
                """)

                # ----------------------------------------------------------
                # 🔎 TABLE 3: extraction_log
                # The detective's notebook — every extraction attempt logged!
                # ----------------------------------------------------------
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS extraction_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        candidate_id INTEGER NOT NULL,
                        field_name TEXT NOT NULL,
                        extraction_method TEXT NOT NULL,
                        extracted_value TEXT,
                        was_successful INTEGER DEFAULT 0,
                        was_overridden INTEGER DEFAULT 0,
                        override_reason TEXT,
                        error_message TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # ----------------------------------------------------------
                # 📊 TABLE 4: schema_version (migration tracking)
                # ----------------------------------------------------------
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY,
                        applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # ----------------------------------------------------------
                # ⚡ INDEXES — Speed up common queries!
                # Like putting your most-used wigs on the TOP shelf! 👑
                # ----------------------------------------------------------
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_raw_candidate 
                    ON raw_extractions(candidate_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_structured_candidate 
                    ON structured_extractions(candidate_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_structured_status 
                    ON structured_extractions(extraction_status)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_log_candidate 
                    ON extraction_log(candidate_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_log_field 
                    ON extraction_log(field_name)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_log_method 
                    ON extraction_log(extraction_method)
                """)
                # ⚡ OPTIMIZATION: Composite index for "get latest extraction" queries
                # Covers the common pattern: WHERE candidate_id = ? ORDER BY created_at DESC LIMIT 1
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_structured_candidate_created
                    ON structured_extractions(candidate_id, created_at DESC)
                """)

                # ⚡ OPTIMIZATION: Same pattern for raw_extractions
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_raw_candidate_timestamp
                    ON raw_extractions(candidate_id, extraction_timestamp DESC)
                """)

                # Record schema version (ignore if already exists)
                conn.execute("""
                    INSERT OR IGNORE INTO schema_version (version)
                    VALUES (?)
                """, (self.SCHEMA_VERSION,))

            logger.info("🏗️ Database schema initialized successfully!")
            
        except sqlite3.Error as e:
            logger.error(f"❌ Schema initialization failed: {e}")
            raise

    # ==========================================================================
    # 💾 CORE DATA OPERATIONS — The Main Event! 🎭
    # ==========================================================================

    def save_raw_extraction(
        self,
        candidate_id: Optional[int],
        folder_path: str,
        filenames: str,
        raw_text: str,
        resume_language: str = "English",
        pdf_page_count: Optional[int] = None
    ) -> Optional[int]:
        """
        📸 Save the RAW extracted text from a resume.
        
        This is the "before" photo — we keep the original text EXACTLY
        as it was extracted from the PDF/DOCX, before any parsing magic.
        
        Args:
            candidate_id:    Numeric ID from the folder name (e.g., 12345)
            folder_path:     Full path to the candidate's folder
            filenames:       Comma-separated list of processed filenames
            raw_text:        The full extracted text from the resume
            resume_language: Detected language ('English', 'Japanese', etc.)
            pdf_page_count:  Number of pages in the PDF (if applicable)
            
        Returns:
            The row ID of the inserted record, or None on failure.
            
        Note:
            Uses INSERT OR REPLACE to handle re-processing gracefully.
            If the same candidate_id + filenames combo exists, it UPDATES
            instead of creating a duplicate. No drama! 🎭
        """
        if not raw_text:
            logger.warning(f"⚠️ No raw text to save for candidate {candidate_id}")
            return None

        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT OR REPLACE INTO raw_extractions 
                    (candidate_id, folder_path, filenames, raw_text, 
                     resume_language, text_length, pdf_page_count,
                     extraction_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    candidate_id,
                    folder_path,
                    filenames,
                    raw_text,
                    resume_language,
                    len(raw_text) if raw_text else 0,
                    pdf_page_count,
                    datetime.datetime.now().isoformat()
                ))
                
                row_id = cursor.lastrowid
                logger.debug(f"📸 Raw text saved for candidate {candidate_id} (ID: {row_id}, {len(raw_text)} chars)")
                return row_id
                
        except sqlite3.Error as e:
            logger.error(f"❌ Failed to save raw extraction for candidate {candidate_id}: {e}")
            return None

    def save_structured_extraction(
        self,
        result: Dict,
        raw_extraction_id: Optional[int] = None,
        extraction_method: str = "Unknown"
    ) -> Optional[int]:
        """
        💃 Save the STRUCTURED extraction results!
        
        This takes the same result dict that process_candidate_folder()
        returns and stores it in the database. The exact same data that
        goes to CSV/JSON also goes here — but now it's QUERYABLE! ⚡
        
        Args:
            result:             The result dict from process_candidate_folder()
            raw_extraction_id:  Links back to the raw_extractions table
            extraction_method:  How it was extracted ('Regex Only', 'Regex + AI')
            
        Returns:
            The row ID of the inserted record, or None on failure.
        """
        candidate_id = result.get("ID")
        if not candidate_id:
            logger.warning("⚠️ Cannot save structured extraction without candidate ID")
            return None

        try:
            with self.get_connection() as conn:
                # 🧹 Clean up Skills/Experience for dual storage
                # Store BOTH the raw string AND a JSON version for flexibility
                skills_raw = result.get("Skills")
                skills_json = self._to_json_safe(skills_raw)

                experience_raw = result.get("Working_Experience")
                experience_json = self._to_json_safe(experience_raw)

                education_raw = result.get("School_University")
                education_json = self._to_json_safe(education_raw)

                now_iso = datetime.datetime.now().isoformat()

                # 🔁 IDEMPOTENT RE-PROCESSING (matches raw_extractions behavior):
                # Before inserting, remove any prior structured rows for this
                # candidate so re-running main.py UPDATES instead of duplicating.
                # We preserve the ORIGINAL created_at and any human review work
                # (reviewed flag + review_notes) from the most recent prior row.
                prior = conn.execute("""
                    SELECT created_at, reviewed, review_notes
                    FROM structured_extractions
                    WHERE candidate_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                """, (candidate_id,)).fetchone()

                if prior is not None:
                    created_at    = prior[0] or now_iso        # keep first-seen timestamp
                    prev_reviewed = prior[1] if prior[1] is not None else 0
                    prev_review_notes = prior[2]
                    deleted = conn.execute(
                        "DELETE FROM structured_extractions WHERE candidate_id = ?",
                        (candidate_id,)
                    ).rowcount
                    logger.info(
                        f"🔁 Replacing {deleted} prior structured row(s) for "
                        f"candidate {candidate_id} (re-extraction)"
                    )
                else:
                    created_at        = now_iso
                    prev_reviewed     = 0
                    prev_review_notes = None

                cursor = conn.execute("""
                    INSERT INTO structured_extractions (
                        candidate_id, raw_extraction_id,
                        name, email, phone, date_of_birth, location,
                        skills_raw, skills_json,
                        experience_raw, experience_json,
                        education_raw, education_json,
                        summary, certifications, languages,
                        projects, achievements, references_info, hobbies,
                        extraction_method, ai_assisted, extraction_status, notes,
                        created_at, updated_at,
                        reviewed, review_notes
                    ) VALUES (
                        ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?,
                        ?, ?
                    )
                """, (
                    candidate_id,
                    raw_extraction_id,
                    result.get("Name"),
                    result.get("Email"),
                    result.get("Phone"),
                    result.get("Date_of_Birth"),
                    result.get("Location"),
                    skills_raw,
                    skills_json,
                    experience_raw,
                    experience_json,
                    education_raw,
                    education_json,
                    result.get("Summary"),
                    result.get("Certifications"),
                    result.get("Languages"),
                    result.get("Projects"),
                    result.get("Achievements"),
                    result.get("References"),
                    result.get("Hobbies"),
                    extraction_method,
                    1 if result.get("AI_Assisted") else 0,
                    result.get("Extraction_Status", "Unknown"),
                    result.get("Notes", ""),
                    created_at,           # preserved from prior row if it existed
                    now_iso,              # updated_at = now
                    prev_reviewed,        # preserve human review flag
                    prev_review_notes     # preserve human review notes
                ))

                row_id = cursor.lastrowid
                logger.debug(f"💃 Structured data saved for candidate {candidate_id} (ID: {row_id})")
                return row_id
                
        except sqlite3.Error as e:
            logger.error(f"❌ Failed to save structured extraction for candidate {candidate_id}: {e}")
            return None

    def dedupe_structured_extractions(self, dry_run: bool = False) -> Dict[str, int]:
        """
        🧹 Remove duplicate rows from structured_extractions.

        Older runs of main.py (before the idempotent-save fix) inserted a NEW
        row every time a candidate was re-processed. This collapses each
        candidate down to a SINGLE row — keeping the most recent extraction
        (highest id) while salvaging any human review work (reviewed flag /
        review_notes) from the rows being deleted.

        Args:
            dry_run: If True, only report what WOULD be removed — change nothing.

        Returns:
            {
              'candidates_with_dupes': int,  # candidates that had > 1 row
              'rows_removed':          int,  # total duplicate rows deleted
              'review_flags_salvaged': int,  # kept-rows that inherited a review flag
              'dry_run':               int,  # 1 if nothing was actually changed
            }
        """
        summary = {
            'candidates_with_dupes': 0,
            'rows_removed':          0,
            'review_flags_salvaged': 0,
            'dry_run':               1 if dry_run else 0,
        }

        try:
            with self.get_connection() as conn:
                dupes = conn.execute("""
                    SELECT candidate_id, COUNT(*) AS n
                    FROM structured_extractions
                    GROUP BY candidate_id
                    HAVING n > 1
                """).fetchall()

                summary['candidates_with_dupes'] = len(dupes)

                for candidate_id, n in dupes:
                    # The survivor: most recent row for this candidate
                    keep_row = conn.execute("""
                        SELECT id, reviewed, review_notes
                        FROM structured_extractions
                        WHERE candidate_id = ?
                        ORDER BY id DESC
                        LIMIT 1
                    """, (candidate_id,)).fetchone()
                    keep_id, keep_reviewed, keep_review_notes = keep_row

                    # Salvage review work from ANY of the soon-to-be-deleted rows
                    salvage = conn.execute("""
                        SELECT reviewed, review_notes
                        FROM structured_extractions
                        WHERE candidate_id = ? AND id != ?
                              AND (reviewed = 1 OR review_notes IS NOT NULL)
                        ORDER BY id DESC
                        LIMIT 1
                    """, (candidate_id, keep_id)).fetchone()

                    if salvage and not (keep_reviewed or keep_review_notes):
                        salvaged_reviewed, salvaged_notes = salvage
                        if not dry_run:
                            conn.execute("""
                                UPDATE structured_extractions
                                SET reviewed = ?, review_notes = ?
                                WHERE id = ?
                            """, (salvaged_reviewed, salvaged_notes, keep_id))
                        summary['review_flags_salvaged'] += 1

                    if not dry_run:
                        removed = conn.execute("""
                            DELETE FROM structured_extractions
                            WHERE candidate_id = ? AND id != ?
                        """, (candidate_id, keep_id)).rowcount
                    else:
                        removed = n - 1  # would-be count

                    summary['rows_removed'] += removed

            verb = "Would remove" if dry_run else "Removed"
            logger.info(
                f"🧹 Dedupe: {verb} {summary['rows_removed']} duplicate row(s) "
                f"across {summary['candidates_with_dupes']} candidate(s); "
                f"{summary['review_flags_salvaged']} review flag(s) salvaged."
            )

        except sqlite3.Error as e:
            logger.error(f"❌ Dedupe failed: {e}")

        return summary

    def log_field_extraction(
        self,
        candidate_id: int,
        field_name: str,
        extraction_method: str,
        extracted_value: Optional[str] = None,
        was_successful: bool = False,
        was_overridden: bool = False,
        override_reason: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> Optional[int]:
        """
        🔎 Log a single field extraction attempt!
        
        This is the DETECTIVE'S NOTEBOOK — every time we try to extract
        a field (name, email, phone, etc.), we log WHAT method we used
        and WHETHER it succeeded. This is GOLD for debugging! 🥇
        
        Args:
            candidate_id:      Which candidate this is for
            field_name:        Which field ('name', 'email', 'phone', etc.)
            extraction_method: How we tried ('regex', 'ai_header', 'ai_deep', 'fallback')
            extracted_value:   What we actually got (truncated for safety)
            was_successful:    Did it produce a usable value?
            was_overridden:    Was this value later replaced by another method?
            override_reason:   Why it was replaced (e.g., 'AI provided better result')
            error_message:     If extraction failed, what went wrong?
            
        Returns:
            The row ID of the log entry, or None on failure.
        """
        try:
            with self.get_connection() as conn:
                # Truncate value for storage safety (don't store megabytes of text!)
                safe_value = str(extracted_value)[:500] if extracted_value else None
                
                cursor = conn.execute("""
                    INSERT INTO extraction_log (
                        candidate_id, field_name, extraction_method,
                        extracted_value, was_successful, was_overridden,
                        override_reason, error_message, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    candidate_id,
                    field_name,
                    extraction_method,
                    safe_value,
                    1 if was_successful else 0,
                    1 if was_overridden else 0,
                    override_reason,
                    error_message,
                    datetime.datetime.now().isoformat()
                ))
                return cursor.lastrowid
                
        except sqlite3.Error as e:
            # Don't let logging failures crash the main extraction!
            # The show must go on! 🎭
            logger.debug(f"⚠️ Failed to log extraction for {field_name}: {e}")
            return None

    def save_extraction(
        self,
        result: Dict,
        raw_text: str,
        folder_path: str,
        extraction_logs: Optional[List[Dict]] = None
    ) -> Dict[str, Optional[int]]:
        """
        🌟 THE ALL-IN-ONE SAVE METHOD! 🌟
        
        This is the method you'll call from process_resumes() in main.py.
        It saves EVERYTHING in one go — raw text, structured data, and logs.
        
        Think of this as the "one-stop glam shop" — walk in messy,
        walk out with everything saved and indexed! 💅
        
        Args:
            result:          The result dict from process_candidate_folder()
            raw_text:        The combined raw text extracted from resume files
            folder_path:     Path to the candidate's folder
            extraction_logs: Optional list of extraction log entries
            
        Returns:
            Dict with IDs: {'raw_id': int, 'structured_id': int, 'log_count': int}
            
        Example:
            # In process_resumes(), right after getting the result:
            result = extractor.process_candidate_folder(candidate_folder_path)
            if result and result.get("ID"):
                db.save_extraction(
                    result=result,
                    raw_text=combined_text,
                    folder_path=candidate_folder_path
                )
        """
        ids = {"raw_id": None, "structured_id": None, "log_count": 0}
        
        candidate_id = result.get("ID")
        if not candidate_id:
            logger.warning("⚠️ Skipping DB save — no candidate ID in result")
            return ids

        # 📸 Step 1: Save raw text
        raw_id = self.save_raw_extraction(
            candidate_id=candidate_id,
            folder_path=folder_path,
            filenames=result.get("Filenames_Processed", ""),
            raw_text=raw_text,
            resume_language=result.get("Language", "English")
        )
        ids["raw_id"] = raw_id

        # 💃 Step 2: Save structured data
        extraction_method = "Regex + AI" if result.get("AI_Assisted") else "Regex Only"
        structured_id = self.save_structured_extraction(
            result=result,
            raw_extraction_id=raw_id,
            extraction_method=extraction_method
        )
        ids["structured_id"] = structured_id

        # 🔎 Step 3: Save extraction logs (if provided)
        if extraction_logs:
            for log_entry in extraction_logs:
                self.log_field_extraction(
                    candidate_id=candidate_id,
                    **log_entry
                )
                ids["log_count"] += 1

        # 🔎 Step 3b: Auto-generate basic field logs from the result
        # Even without explicit logs, we can track which fields succeeded!
        tracked_fields = {
            "name": result.get("Name"),
            "email": result.get("Email"),
            "phone": result.get("Phone"),
            "date_of_birth": result.get("Date_of_Birth"),
            "skills": result.get("Skills"),
            "experience": result.get("Working_Experience"),
            "education": result.get("School_University"),
            "location": result.get("Location"),
        }
        
        for field_name, value in tracked_fields.items():
            self.log_field_extraction(
                candidate_id=candidate_id,
                field_name=field_name,
                extraction_method=extraction_method,
                extracted_value=value,
                was_successful=value is not None and str(value).strip() != ""
            )
            ids["log_count"] += 1

        logger.info(
            f"💾 Candidate {candidate_id} saved to DB "
            f"(raw_id={raw_id}, structured_id={structured_id}, "
            f"logs={ids['log_count']})"
        )
        return ids

    # ==========================================================================
    # 🔍 DIAGNOSTIC QUERIES — The Investigation Suite! 🕵️‍♀️
    # ==========================================================================

    def get_missing_fields_report(self) -> List[Dict]:
        """
        🚨 Find all candidates with missing critical fields!
        
        This is the "Who showed up to the ball without their shoes?" report! 👠
        
        Returns:
            List of dicts with candidate_id, name, and which fields are missing.
        """
        try:
            cursor = self._connection.execute("""
                SELECT 
                    candidate_id,
                    name,
                    CASE WHEN name IS NULL OR name = '' THEN 1 ELSE 0 END as missing_name,
                    CASE WHEN email IS NULL OR email = '' THEN 1 ELSE 0 END as missing_email,
                    CASE WHEN phone IS NULL OR phone = '' THEN 1 ELSE 0 END as missing_phone,
                    CASE WHEN date_of_birth IS NULL OR date_of_birth = '' THEN 1 ELSE 0 END as missing_dob,
                    CASE WHEN skills_raw IS NULL OR skills_raw = '' THEN 1 ELSE 0 END as missing_skills,
                    CASE WHEN experience_raw IS NULL OR experience_raw = '' THEN 1 ELSE 0 END as missing_experience,
                    CASE WHEN education_raw IS NULL OR education_raw = '' THEN 1 ELSE 0 END as missing_education,
                    extraction_status,
                    ai_assisted
                FROM structured_extractions
                WHERE name IS NULL OR email IS NULL OR phone IS NULL 
                      OR date_of_birth IS NULL
                ORDER BY candidate_id
            """)
            
            results = []
            for row in cursor.fetchall():
                missing = []
                if row["missing_name"]:     missing.append("Name")
                if row["missing_email"]:    missing.append("Email")
                if row["missing_phone"]:    missing.append("Phone")
                if row["missing_dob"]:      missing.append("Date_of_Birth")
                if row["missing_skills"]:   missing.append("Skills")
                if row["missing_experience"]: missing.append("Experience")
                if row["missing_education"]:  missing.append("Education")
                
                results.append({
                    "candidate_id": row["candidate_id"],
                    "name": row["name"] or "UNKNOWN",
                    "missing_fields": missing,
                    "missing_count": len(missing),
                    "extraction_status": row["extraction_status"],
                    "ai_assisted": bool(row["ai_assisted"])
                })
            
            return results
            
        except sqlite3.Error as e:
            logger.error(f"❌ Missing fields query failed: {e}")
            return []

    def get_wrong_field_suspects(self) -> List[Dict]:
        """
        🕵️ Find candidates where data might be in the WRONG field!
        
        This checks for suspicious patterns like:
        - Email addresses in the name field
        - Phone numbers in the email field
        - Locations in the name field
        
        The plot twist detector! 🎭
        
        Returns:
            List of candidates with suspicious field values.
        """
        try:
            cursor = self._connection.execute("""
                SELECT candidate_id, name, email, phone, location
                FROM structured_extractions
                WHERE 
                    -- Email pattern in name field 
                    (name LIKE '%@%.%')
                    -- Phone digits in email field
                    OR (email IS NOT NULL AND email NOT LIKE '%@%')
                    -- Suspiciously long name (might contain address)
                    OR (LENGTH(name) > 50)
                    -- Name contains numbers (might be ID or phone)
                    OR (name GLOB '*[0-9][0-9][0-9]*')
                    -- Phone field has @ (email in phone)
                    OR (phone LIKE '%@%')
                ORDER BY candidate_id
            """)
            
            results = []
            for row in cursor.fetchall():
                issues = []
                name = row["name"] or ""
                email = row["email"] or ""
                phone = row["phone"] or ""
                
                if "@" in name:
                    issues.append("Email found in Name field")
                if email and "@" not in email:
                    issues.append("Non-email value in Email field")
                if len(name) > 50:
                    issues.append(f"Name suspiciously long ({len(name)} chars)")
                if name and any(c.isdigit() for c in name) and sum(c.isdigit() for c in name) > 2:
                    issues.append("Name contains multiple digits")
                if "@" in phone:
                    issues.append("Email found in Phone field")
                
                if issues:
                    results.append({
                        "candidate_id": row["candidate_id"],
                        "name": name[:80],
                        "email": email[:80],
                        "phone": phone[:30],
                        "issues": issues
                    })
            
            return results
            
        except sqlite3.Error as e:
            logger.error(f"❌ Wrong field suspects query failed: {e}")
            return []

    def get_raw_text_for_candidate(self, candidate_id: int) -> Optional[str]:
        """
        📜 Retrieve the original raw text for a specific candidate.
        
        When you find a wrong extraction, THIS is how you go back
        and check "what did the resume ACTUALLY say?" 🔍
        
        Args:
            candidate_id: The candidate's numeric ID
            
        Returns:
            The raw text string, or None if not found.
        """
        try:
            cursor = self._connection.execute("""
                SELECT raw_text FROM raw_extractions
                WHERE candidate_id = ?
                ORDER BY extraction_timestamp DESC
                LIMIT 1
            """, (candidate_id,))
            
            row = cursor.fetchone()
            return row["raw_text"] if row else None
            
        except sqlite3.Error as e:
            logger.error(f"❌ Failed to retrieve raw text for candidate {candidate_id}: {e}")
            return None

    def get_extraction_diagnostics(self, candidate_id: int) -> Dict:
        """
        🩺 Full diagnostic report for a single candidate!
        
        This is the FULL MEDICAL EXAM — everything we know about
        how this candidate's data was extracted. Perfect for debugging
        "why is this field wrong?" situations! 🏥
        
        Args:
            candidate_id: The candidate's numeric ID
            
        Returns:
            Dict with raw_info, structured_data, extraction_history
        """
        diagnostics = {
            "candidate_id": candidate_id,
            "raw_info": None,
            "structured_data": None,
            "extraction_history": [],
            "field_attempts": {}
        }

        try:
            # 📸 Get raw extraction info (metadata, not full text)
            cursor = self._connection.execute("""
                SELECT id, candidate_id, folder_path, filenames,
                       text_length, resume_language, extraction_timestamp
                FROM raw_extractions
                WHERE candidate_id = ?
                ORDER BY extraction_timestamp DESC
                LIMIT 1
            """, (candidate_id,))
            
            row = cursor.fetchone()
            if row:
                diagnostics["raw_info"] = dict(row)

            # 💃 Get structured extraction
            cursor = self._connection.execute("""
                SELECT * FROM structured_extractions
                WHERE candidate_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (candidate_id,))
            
            row = cursor.fetchone()
            if row:
                diagnostics["structured_data"] = dict(row)

            # 🔎 Get all extraction log entries
            cursor = self._connection.execute("""
                SELECT field_name, extraction_method, extracted_value,
                       was_successful, was_overridden, override_reason,
                       error_message, timestamp
                FROM extraction_log
                WHERE candidate_id = ?
                ORDER BY timestamp ASC
            """, (candidate_id,))
            
            for row in cursor.fetchall():
                entry = dict(row)
                diagnostics["extraction_history"].append(entry)
                
                # Group by field for easy analysis
                field = entry["field_name"]
                if field not in diagnostics["field_attempts"]:
                    diagnostics["field_attempts"][field] = []
                diagnostics["field_attempts"][field].append(entry)

            return diagnostics
            
        except sqlite3.Error as e:
            logger.error(f"❌ Diagnostics query failed for candidate {candidate_id}: {e}")
            return diagnostics

    def diagnose_missing_data(self, candidate_id: int) -> str:
        """
        🩺💬 Human-readable diagnosis for WHY data is missing!
        
        This is the doctor's summary — instead of raw data, you get
        a clear explanation of what went wrong and where. 📋
        
        Args:
            candidate_id: The candidate's numeric ID
            
        Returns:
            A formatted string with the diagnosis.
        """
        diag = self.get_extraction_diagnostics(candidate_id)
        lines = []
        lines.append(f"🩺 DIAGNOSIS FOR CANDIDATE {candidate_id}")
        lines.append("=" * 50)
        
        # Check if raw text exists
        raw = diag.get("raw_info")
        if not raw:
            lines.append("❌ NO RAW TEXT FOUND — PDF extraction may have failed entirely!")
            return "\n".join(lines)
        
        lines.append(f"📸 Raw text: {raw.get('text_length', 0)} characters extracted")
        lines.append(f"📁 Source: {raw.get('filenames', 'Unknown')}")
        
        # Check structured data
        structured = diag.get("structured_data")
        if not structured:
            lines.append("❌ NO STRUCTURED DATA — Extraction pipeline failed after text extraction!")
            return "\n".join(lines)
        
        lines.append(f"📊 Status: {structured.get('extraction_status', 'Unknown')}")
        lines.append(f"🤖 AI Assisted: {'Yes' if structured.get('ai_assisted') else 'No'}")
        
        # Check each critical field
        critical_fields = {
            "name": structured.get("name"),
            "email": structured.get("email"),
            "phone": structured.get("phone"),
            "date_of_birth": structured.get("date_of_birth"),
            "skills": structured.get("skills_raw"),
            "experience": structured.get("experience_raw"),
            "education": structured.get("education_raw"),
        }
        
        lines.append("\n📋 FIELD STATUS:")
        for field, value in critical_fields.items():
            status = "✅" if value else "❌ MISSING"
            preview = str(value)[:60] + "..." if value and len(str(value)) > 60 else value
            lines.append(f"   {status} {field}: {preview or 'None'}")
        
        # Check if raw text HAD the missing section keywords
        raw_text = self.get_raw_text_for_candidate(candidate_id)
        if raw_text:
            missing_fields = [f for f, v in critical_fields.items() if not v]
            if missing_fields:
                lines.append("\n🔍 RAW TEXT ANALYSIS (checking if sections exist):")
                
                section_keywords = {
                    "skills": ["SKILLS", "COMPETENCIES", "EXPERTISE", "PROFICIENCIES"],
                    "experience": ["EXPERIENCE", "EMPLOYMENT", "WORK HISTORY", "CAREER"],
                    "education": ["EDUCATION", "QUALIFICATIONS", "ACADEMIC", "UNIVERSITY"],
                    "email": ["@", "EMAIL", "E-MAIL"],
                    "phone": ["PHONE", "MOBILE", "CONTACT", "TEL"],
                    "date_of_birth": ["DOB", "DATE OF BIRTH", "BIRTHDAY", "BORN"],
                    "name": []  # Name is usually at the top, hard to keyword-search
                }
                
                raw_upper = raw_text.upper()
                for field in missing_fields:
                    keywords = section_keywords.get(field, [])
                    found_keywords = [kw for kw in keywords if kw in raw_upper]
                    if found_keywords:
                        lines.append(
                            f"   ⚠️ {field}: Keywords {found_keywords} FOUND in raw text! "
                            f"→ Extraction FAILED to capture this section"
                        )
                    else:
                        lines.append(
                            f"   ℹ️ {field}: No relevant keywords found in raw text "
                            f"→ Resume may genuinely lack this section"
                        )
        
        return "\n".join(lines)

    # ==========================================================================
    # 📊 STATISTICS & REPORTING
    # ==========================================================================

    def get_database_stats(self) -> Dict:
        """
        📊 Get a comprehensive summary of the database!
        
        Returns:
            Dict with counts, status breakdowns, and quality metrics.
        """
        stats = {}
        
        try:
            # Total counts
            for table in ["raw_extractions", "structured_extractions", "extraction_log"]:
                cursor = self._connection.execute(f"SELECT COUNT(*) as cnt FROM {table}")
                stats[f"total_{table}"] = cursor.fetchone()["cnt"]
            
            # Extraction status breakdown
            cursor = self._connection.execute("""
                SELECT extraction_status, COUNT(*) as cnt
                FROM structured_extractions
                GROUP BY extraction_status
                ORDER BY cnt DESC
            """)
            stats["status_breakdown"] = {row["extraction_status"]: row["cnt"] for row in cursor.fetchall()}
            
            # AI usage stats
            cursor = self._connection.execute("""
                SELECT 
                    SUM(CASE WHEN ai_assisted = 1 THEN 1 ELSE 0 END) as ai_count,
                    SUM(CASE WHEN ai_assisted = 0 THEN 1 ELSE 0 END) as regex_count
                FROM structured_extractions
            """)
            row = cursor.fetchone()
            stats["ai_assisted_count"] = row["ai_count"] or 0
            stats["regex_only_count"] = row["regex_count"] or 0
            
            # Field completeness (what % of candidates have each field?)
            fields = ["name", "email", "phone", "date_of_birth", 
                       "skills_raw", "experience_raw", "education_raw"]
            total = stats.get("total_structured_extractions", 0)
            
            completeness = {}
            for field in fields:
                if total > 0:
                    cursor = self._connection.execute(f"""
                        SELECT COUNT(*) as cnt FROM structured_extractions
                        WHERE {field} IS NOT NULL AND {field} != ''
                    """)
                    filled = cursor.fetchone()["cnt"]
                    completeness[field] = round((filled / total) * 100, 1)
                else:
                    completeness[field] = 0.0
            stats["field_completeness_pct"] = completeness
            
            # Extraction method success rates from logs
            cursor = self._connection.execute("""
                SELECT 
                    extraction_method,
                    COUNT(*) as total,
                    SUM(was_successful) as successes
                FROM extraction_log
                GROUP BY extraction_method
            """)
            method_stats = {}
            for row in cursor.fetchall():
                total_attempts = row["total"]
                successes = row["successes"] or 0
                rate = round((successes / total_attempts) * 100, 1) if total_attempts > 0 else 0.0
                method_stats[row["extraction_method"]] = {
                    "total": total_attempts,
                    "successes": successes,
                    "success_rate": rate
                }
            stats["method_success_rates"] = method_stats
            
            return stats
            
        except sqlite3.Error as e:
            logger.error(f"❌ Stats query failed: {e}")
            return stats

    def print_stats_report(self):
        """
        📊✨ Print a GORGEOUS stats report to console!
        
        This is the "after-party recap" — all the highlights
        from the extraction run! 🎉
        """
        stats = self.get_database_stats()
        
        print("\n" + "╔" + "═" * 68 + "╗")
        print("║" + "  🗄️✨ DATABASE STATS REPORT ✨🗄️  ".center(68) + "║")
        print("╠" + "═" * 68 + "╣")
        
        # Record counts
        print("║" + f"  📸 Raw Extractions:        {stats.get('total_raw_extractions', 0):>6}".ljust(68) + "║")
        print("║" + f"  💃 Structured Extractions:  {stats.get('total_structured_extractions', 0):>6}".ljust(68) + "║")
        print("║" + f"  🔎 Extraction Log Entries:  {stats.get('total_extraction_log', 0):>6}".ljust(68) + "║")
        
        # Status breakdown
        status = stats.get("status_breakdown", {})
        if status:
            print("║" + " " * 68 + "║")
            print("║" + "  📋 EXTRACTION STATUS BREAKDOWN:".ljust(68) + "║")
            for s, count in status.items():
                emoji = {"Complete": "✅", "Success": "🟢", "Partial": "🟡", "Failed": "🔴"}.get(s, "⚪")
                print("║" + f"      {emoji} {s}: {count}".ljust(68) + "║")
        
        # AI vs Regex
        ai_count = stats.get("ai_assisted_count", 0)
        regex_count = stats.get("regex_only_count", 0)
        total = ai_count + regex_count
        if total > 0:
            print("║" + " " * 68 + "║")
            print("║" + "  🤖 EXTRACTION METHOD:".ljust(68) + "║")
            print("║" + f"      AI Assisted: {ai_count} ({ai_count/total*100:.1f}%)".ljust(68) + "║")
            print("║" + f"      Regex Only:  {regex_count} ({regex_count/total*100:.1f}%)".ljust(68) + "║")
        
        # Field completeness
        completeness = stats.get("field_completeness_pct", {})
        if completeness:
            print("║" + " " * 68 + "║")
            print("║" + "  🎯 FIELD COMPLETENESS:".ljust(68) + "║")
            for field, pct in completeness.items():
                bar_len = int(pct / 5)  # Scale to 20 chars max
                bar = "█" * bar_len + "░" * (20 - bar_len)
                # Clean up field name for display
                display_name = field.replace("_raw", "").replace("_", " ").title()
                print("║" + f"      {display_name:15s} {bar} {pct:5.1f}%".ljust(68) + "║")
        
        print("╚" + "═" * 68 + "╝")

    # ==========================================================================
    # 🔄 COMPARISON & RE-PROCESSING HELPERS
    # ==========================================================================

    def get_candidates_for_reprocessing(
        self, 
        status_filter: Optional[str] = None,
        missing_field: Optional[str] = None
    ) -> List[int]:
        """
        🔄 Get list of candidate IDs that need re-processing!
        
        Useful after you've improved your regex patterns or AI prompts
        and want to re-extract specific problem candidates.
        
        Args:
            status_filter:  Filter by extraction status ('Failed', 'Partial', etc.)
            missing_field:  Filter by a specific missing field ('email', 'phone', etc.)
            
        Returns:
            List of candidate_id integers.
        """
        try:
            conditions = []
            params = []
            
            if status_filter:
                conditions.append("extraction_status = ?")
                params.append(status_filter)
            
            if missing_field:
                # Map common field names to column names
                field_map = {
                    "name": "name", "email": "email", "phone": "phone",
                    "dob": "date_of_birth", "date_of_birth": "date_of_birth",
                    "skills": "skills_raw", "experience": "experience_raw",
                    "education": "education_raw"
                }
                col = field_map.get(missing_field.lower(), missing_field)
                conditions.append(f"({col} IS NULL OR {col} = '')")
            
            where = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            cursor = self._connection.execute(f"""
                SELECT DISTINCT candidate_id 
                FROM structured_extractions
                {where}
                ORDER BY candidate_id
            """, params)
            
            return [row["candidate_id"] for row in cursor.fetchall()]
            
        except sqlite3.Error as e:
            logger.error(f"❌ Reprocessing query failed: {e}")
            return []

    def compare_extractions(self, candidate_id: int) -> List[Dict]:
        """
        🔀 Compare ALL extraction attempts for a candidate!
        
        If a candidate was processed multiple times (e.g., after
        code improvements), this shows how the results changed.
        Like a before-and-after montage! 🎬
        
        Args:
            candidate_id: The candidate's numeric ID
            
        Returns:
            List of all structured extractions for this candidate,
            ordered by creation time.
        """
        try:
            cursor = self._connection.execute("""
                SELECT * FROM structured_extractions
                WHERE candidate_id = ?
                ORDER BY created_at ASC
            """, (candidate_id,))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except sqlite3.Error as e:
            logger.error(f"❌ Comparison query failed for candidate {candidate_id}: {e}")
            return []

    def search_raw_text(self, keyword: str, limit: int = 20) -> List[Dict]:
        """
        🔍 Search through raw resume texts for a keyword!
        
        Great for finding patterns like "how many resumes mention Docker?"
        or debugging specific extraction failures.
        
        Args:
            keyword: The text to search for (case-insensitive)
            limit:   Maximum results to return
            
        Returns:
            List of matching candidates with snippet previews.
        """
        try:
            cursor = self._connection.execute("""
                SELECT 
                    r.candidate_id,
                    r.filenames,
                    r.text_length,
                    s.name,
                    s.extraction_status,
                    r.raw_text
                FROM raw_extractions r
                LEFT JOIN structured_extractions s 
                    ON r.candidate_id = s.candidate_id
                WHERE r.raw_text LIKE ?
                ORDER BY r.candidate_id
                LIMIT ?
            """, (f"%{keyword}%", limit))
            
            results = []
            for row in cursor.fetchall():
                # Find the keyword in context (show surrounding text)
                raw = row["raw_text"] or ""
                idx = raw.lower().find(keyword.lower())
                snippet = ""
                if idx >= 0:
                    start = max(0, idx - 50)
                    end = min(len(raw), idx + len(keyword) + 50)
                    snippet = "..." + raw[start:end] + "..."
                
                results.append({
                    "candidate_id": row["candidate_id"],
                    "name": row["name"] or "UNKNOWN",
                    "filenames": row["filenames"],
                    "status": row["extraction_status"],
                    "snippet": snippet
                })
            
            return results
            
        except sqlite3.Error as e:
            logger.error(f"❌ Raw text search failed: {e}")
            return []

    # ==========================================================================
    # 🧹 MAINTENANCE UTILITIES
    # ==========================================================================

    def mark_as_reviewed(self, candidate_id: int, notes: str = ""):
        """
        ✅ Mark a candidate's extraction as manually reviewed.
        
        After a human verifies the data is correct (or corrects it),
        mark it so we know it's been quality-checked.
        
        Args:
            candidate_id: The candidate's numeric ID
            notes:        Optional reviewer notes
        """
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    UPDATE structured_extractions
                    SET reviewed = 1, 
                        review_notes = ?,
                        updated_at = ?
                    WHERE candidate_id = ?
                """, (notes, datetime.datetime.now().isoformat(), candidate_id))
                
                logger.info(f"✅ Candidate {candidate_id} marked as reviewed.")
                
        except sqlite3.Error as e:
            logger.error(f"❌ Failed to mark candidate {candidate_id} as reviewed: {e}")

    def update_field(self, candidate_id: int, field_name: str, new_value: str):
        """
        ✏️ Manually correct a specific field for a candidate.
        
        When you find a wrong extraction, use this to fix it AND
        log the correction for future pattern learning!
        
        Args:
            candidate_id: The candidate's numeric ID
            field_name:   Column name to update (e.g., 'name', 'email')
            new_value:    The corrected value
        """
        # Map user-friendly names to actual column names
        field_map = {
            "name": "name", "email": "email", "phone": "phone",
            "dob": "date_of_birth", "date_of_birth": "date_of_birth",
            "location": "location", "nationality": "nationality",
            "skills": "skills_raw", "experience": "experience_raw",
            "education": "education_raw", "summary": "summary",
        }
        
        col = field_map.get(field_name.lower())
        if not col:
            logger.error(f"❌ Unknown field: {field_name}")
            return
        
        try:
            # Get the old value first (for logging)
            cursor = self._connection.execute(f"""
                SELECT {col} FROM structured_extractions
                WHERE candidate_id = ?
                ORDER BY created_at DESC LIMIT 1
            """, (candidate_id,))
            row = cursor.fetchone()
            old_value = row[col] if row else None
            
            with self.get_connection() as conn:
                conn.execute(f"""
                    UPDATE structured_extractions
                    SET {col} = ?,
                        updated_at = ?,
                        review_notes = COALESCE(review_notes, '') || ?
                    WHERE candidate_id = ?
                """, (
                    new_value,
                    datetime.datetime.now().isoformat(),
                    f"\n[CORRECTED] {field_name}: '{old_value}' → '{new_value}'",
                    candidate_id
                ))
                
                # Log the correction
                self.log_field_extraction(
                    candidate_id=candidate_id,
                    field_name=field_name,
                    extraction_method="manual_correction",
                    extracted_value=new_value,
                    was_successful=True,
                    was_overridden=True,
                    override_reason=f"Manual correction from '{old_value}'"
                )
                
                logger.info(f"✏️ Updated {field_name} for candidate {candidate_id}: '{old_value}' → '{new_value}'")
                
        except sqlite3.Error as e:
            logger.error(f"❌ Failed to update {field_name} for candidate {candidate_id}: {e}")

    def delete_candidate(self, candidate_id: int):
        """
        🗑️ Delete ALL data for a candidate (raw, structured, and logs).
        
        Use with caution, darling! This is permanent! 💀
        
        Args:
            candidate_id: The candidate's numeric ID
        """
        try:
            with self.get_connection() as conn:
                for table in ["extraction_log", "structured_extractions", "raw_extractions"]:
                    conn.execute(f"DELETE FROM {table} WHERE candidate_id = ?", (candidate_id,))
                
                logger.info(f"🗑️ All data deleted for candidate {candidate_id}")
                
        except sqlite3.Error as e:
            logger.error(f"❌ Failed to delete candidate {candidate_id}: {e}")

    def export_to_json(self, output_path: str = "db_export.json") -> str:
        """
        📤 Export all structured extractions to JSON!
        
        Useful for backup or for sharing data with other tools.
        
        Args:
            output_path: Where to save the JSON file
            
        Returns:
            The path to the saved file.
        """
        try:
            cursor = self._connection.execute("""
                SELECT s.*, r.text_length, r.resume_language, r.filenames
                FROM structured_extractions s
                LEFT JOIN raw_extractions r ON s.raw_extraction_id = r.id
                ORDER BY s.candidate_id
            """)
            
            results = [dict(row) for row in cursor.fetchall()]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"📤 Exported {len(results)} records to {output_path}")
            return output_path
            
        except (sqlite3.Error, IOError) as e:
            logger.error(f"❌ Export failed: {e}")
            return ""

    # ==========================================================================
    # 🔧 INTERNAL HELPERS
    # ==========================================================================

    def _to_json_safe(self, value) -> Optional[str]:
        """
        🔧 Safely convert a value to JSON string for dual storage.
        
        Handles pipe-separated strings (from _format_skills_for_export),
        plain strings, lists, dicts, and None values.
        """
        if value is None:
            return None
        
        # If it's already a valid JSON string, return as-is
        if isinstance(value, str):
            # Check if it's a pipe-separated list (our export format)
            if " | " in value:
                items = [item.strip() for item in value.split(" | ") if item.strip()]
                return json.dumps(items, ensure_ascii=False)
            
            # Check if it's already JSON
            try:
                json.loads(value)
                return value  # Already valid JSON
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Plain string — wrap in JSON
            return json.dumps(value, ensure_ascii=False)
        
        # Lists, dicts, etc. — serialize directly
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return json.dumps(str(value))

    def __enter__(self):
        """Support 'with' statement for clean resource management."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Auto-close on exiting 'with' block."""
        self.close()
        return False  # Don't suppress exceptions

    def __del__(self):
        """Safety net — close connection if garbage collected."""
        self.close()