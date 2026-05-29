"""
test_sg_normalizer.py
=====================
✨ FAIRY CODEMOTHER'S TEST SUITE — No Garbage Gets Past The Velvet Rope! ✨

Run with:
    pytest test_sg_normalizer.py -v

The tests build a small in-memory / temp-file database so they have zero
dependency on the real sg_normalizer.db being present.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from sg_normalizer import SGInstitutionNormalizer


# ══════════════════════════════════════════════════════════════════
#  FIXTURES — build a tiny but realistic test DB every run
# ══════════════════════════════════════════════════════════════════

DDL = """
CREATE TABLE institutions (
    institution_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name        TEXT NOT NULL UNIQUE,
    short_code            TEXT NOT NULL,
    is_partner_institution INTEGER NOT NULL DEFAULT 0,
    country               TEXT NOT NULL DEFAULT 'SG'
);
CREATE TABLE institution_aliases (
    alias_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    institution_id INTEGER NOT NULL REFERENCES institutions(institution_id),
    alias_text     TEXT NOT NULL UNIQUE
);
CREATE TABLE degrees (
    degree_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name     TEXT NOT NULL,
    base_name          TEXT NOT NULL,
    institution_id     INTEGER NOT NULL REFERENCES institutions(institution_id),
    has_honours        INTEGER NOT NULL DEFAULT 0,
    program_duration   TEXT,
    full_original_name TEXT NOT NULL UNIQUE
);
CREATE INDEX idx_aliases_text ON institution_aliases(alias_text);
CREATE INDEX idx_degrees_base ON degrees(base_name);
"""

INSTITUTIONS = [
    (1, "Nanyang Technological University", "NTU",  0, "SG"),
    (2, "National University of Singapore", "NUS",  0, "SG"),
    (3, "Singapore Management University",  "SMU",  0, "SG"),
    (4, "DigiPen Institute of Technology",  "DigiPen", 1, "US"),
]

ALIASES = [
    (1, "ntu"),
    (1, "nanyang tech"),
    (1, "nanyang technological university"),
    (2, "nus"),
    (2, "national university of singapore"),
    (2, "national u"),
    (3, "smu"),
    (3, "singapore management university"),
    (4, "digipen"),
    (4, "digipen institute of technology"),
]

DEGREES = [
    # (canonical, base_name, inst_id, has_honours, duration, full_original)
    ("Computer Science", "Computer Science", 1, 0, None,
     "Computer Science"),
    ("Computer Science", "Computer Science", 1, 1, None,
     "Computer Science with Honours"),
    ("Accountancy", "Accountancy", 3, 0, None,
     "Accountancy"),
    ("Accountancy", "Accountancy", 3, 0, "4-year",
     "Accountancy (4-year programme)"),
    ("Accountancy", "Accountancy", 3, 1, "4-year",
     "Accountancy (4-year programme) Cum Laude and above"),
    ("Business Administration", "Business Administration", 2, 0, None,
     "Business Administration"),
    ("Bachelor of Engineering", "Bachelor of Engineering", 1, 0, None,
     "Bachelor of Engineering"),
    ("Bachelor of Engineering", "Bachelor of Engineering", 1, 1, None,
     "Bachelor of Engineering with Honours"),
]


@pytest.fixture(scope="module")
def db_path(tmp_path_factory) -> Path:
    """Build a temp SQLite DB and return its path."""
    p = tmp_path_factory.mktemp("data") / "test_sg.db"
    con = sqlite3.connect(p)
    con.executescript(DDL)

    con.executemany(
        "INSERT INTO institutions (institution_id, canonical_name, short_code, "
        "is_partner_institution, country) VALUES (?,?,?,?,?)",
        INSTITUTIONS,
    )
    con.executemany(
        "INSERT INTO institution_aliases (institution_id, alias_text) VALUES (?,?)",
        ALIASES,
    )
    con.executemany(
        "INSERT INTO degrees (canonical_name, base_name, institution_id, has_honours, "
        "program_duration, full_original_name) VALUES (?,?,?,?,?,?)",
        DEGREES,
    )
    con.commit()
    con.close()
    return p


@pytest.fixture(scope="module")
def norm(db_path) -> SGInstitutionNormalizer:
    return SGInstitutionNormalizer(db_path)


# ══════════════════════════════════════════════════════════════════
#  normalize_institution — INSTITUTION TESTS
# ══════════════════════════════════════════════════════════════════

class TestNormalizeInstitution:

    def test_exact_canonical_match(self, norm):
        """Full canonical name → confidence 1.0"""
        r = norm.normalize_institution("Nanyang Technological University")
        assert r is not None
        assert r["canonical_name"] == "Nanyang Technological University"
        assert r["short_code"] == "NTU"
        assert r["confidence"] == 1.0
        assert r["is_partner"] is False

    def test_exact_alias_short_code(self, norm):
        """Short code alias like 'ntu' → confidence 0.95"""
        r = norm.normalize_institution("NTU")
        assert r is not None
        assert r["short_code"] == "NTU"
        assert r["confidence"] == 0.95

    def test_exact_alias_partial_name(self, norm):
        """'nanyang tech' alias → NTU"""
        r = norm.normalize_institution("nanyang tech")
        assert r is not None
        assert r["short_code"] == "NTU"

    def test_alias_national_u(self, norm):
        """'national u' → NUS"""
        r = norm.normalize_institution("national u")
        assert r is not None
        assert r["short_code"] == "NUS"

    def test_fuzzy_match_typo(self, norm):
        """Slight typo should still fuzzy-match above threshold."""
        r = norm.normalize_institution("Nanyang Technological Universty")  # typo
        assert r is not None
        assert r["short_code"] == "NTU"
        assert r["confidence"] < 1.0

    def test_fuzzy_match_digipen(self, norm):
        """'digipen' → DigiPen (partner institution)"""
        r = norm.normalize_institution("digipen")
        assert r is not None
        assert r["short_code"] == "DigiPen"
        assert r["is_partner"] is True
        assert r["country"] == "US"

    def test_partner_institution_flag(self, norm):
        """DigiPen should be flagged as a partner."""
        r = norm.normalize_institution("DigiPen Institute of Technology")
        assert r is not None
        assert r["is_partner"] is True

    def test_no_match_returns_none(self, norm):
        """Completely unknown string → None"""
        r = norm.normalize_institution("Hogwarts School of Witchcraft")
        assert r is None

    def test_empty_string_returns_none(self, norm):
        r = norm.normalize_institution("")
        assert r is None

    def test_none_input_returns_none(self, norm):
        r = norm.normalize_institution(None)  # type: ignore[arg-type]
        assert r is None

    def test_whitespace_only_returns_none(self, norm):
        r = norm.normalize_institution("   ")
        assert r is None

    def test_mixed_case(self, norm):
        """Case should not matter."""
        r = norm.normalize_institution("nUs")
        assert r is not None
        assert r["short_code"] == "NUS"

    def test_extra_punctuation(self, norm):
        """Extra punctuation stripped during normalisation."""
        r = norm.normalize_institution("N.U.S.")
        # Fuzzy or normalised match — should not crash
        # (may or may not match depending on rapidfuzz score; test graceful return)
        assert r is None or isinstance(r, dict)


# ══════════════════════════════════════════════════════════════════
#  normalize_degree — DEGREE TESTS
# ══════════════════════════════════════════════════════════════════

class TestNormalizeDegree:

    def test_exact_base_match(self, norm):
        """Plain degree name matches directly."""
        r = norm.normalize_degree("Computer Science")
        assert r is not None
        assert r["canonical_name"] == "Computer Science"
        assert r["has_honours"] is False

    def test_honours_stripped_base_matches(self, norm):
        """'Computer Science with Honours' → base='Computer Science', has_honours=True"""
        r = norm.normalize_degree("Computer Science with Honours")
        assert r is not None
        assert r["canonical_name"] == "Computer Science"
        assert r["has_honours"] is True

    def test_cum_laude_stripped(self, norm):
        """'Accountancy Cum Laude and above' → base='Accountancy'"""
        r = norm.normalize_degree("Accountancy Cum Laude and above")
        assert r is not None
        assert r["canonical_name"] == "Accountancy"
        assert r["has_honours"] is True

    def test_duration_extracted(self, norm):
        """'Accountancy (4-year programme)' → program_duration='4-year'"""
        r = norm.normalize_degree("Accountancy (4-year programme)")
        assert r is not None
        assert r["program_duration"] == "4-year"

    def test_honours_and_duration_together(self, norm):
        r = norm.normalize_degree("Accountancy (4-year programme) Cum Laude and above")
        assert r is not None
        assert r["canonical_name"] == "Accountancy"
        assert r["has_honours"] is True
        assert r["program_duration"] == "4-year"

    def test_hons_parenthesis(self, norm):
        """'Bachelor of Accountancy (Hons)' → has_honours=True"""
        r = norm.normalize_degree("Bachelor of Engineering (Hons)")
        assert r is not None
        assert r["has_honours"] is True

    def test_institution_hint_narrows_match(self, norm):
        """Same degree at different unis — hint should prefer NTU result."""
        r = norm.normalize_degree("Computer Science", institution_hint="NTU")
        assert r is not None
        assert r["institution"]["short_code"] == "NTU"

    def test_fuzzy_degree_match(self, norm):
        """Slight variation should fuzzy-match."""
        r = norm.normalize_degree("Busines Administration")   # typo
        assert r is not None
        assert r["canonical_name"] == "Business Administration"
        assert r["confidence"] < 1.0

    def test_no_match_returns_none(self, norm):
        r = norm.normalize_degree("Advanced Wizardry and Dark Arts")
        assert r is None

    def test_empty_degree_returns_none(self, norm):
        r = norm.normalize_degree("")
        assert r is None

    def test_none_degree_returns_none(self, norm):
        r = norm.normalize_degree(None)  # type: ignore[arg-type]
        assert r is None

    def test_institution_in_result(self, norm):
        """Result should carry institution details."""
        r = norm.normalize_degree("Accountancy", institution_hint="SMU")
        assert r is not None
        assert r["institution"] is not None
        assert r["institution"]["short_code"] == "SMU"


# ══════════════════════════════════════════════════════════════════
#  bulk_normalize_education
# ══════════════════════════════════════════════════════════════════

class TestBulkNormalize:

    def test_basic_bulk(self, norm):
        records = [
            {"institution": "NTU", "degree": "Computer Science"},
            {"institution": "SMU", "degree": "Accountancy"},
        ]
        result = norm.bulk_normalize_education(records)
        assert len(result) == 2
        assert result[0]["institution_normalized"]["short_code"] == "NTU"
        assert result[0]["degree_normalized"]["canonical_name"] == "Computer Science"
        assert result[1]["institution_normalized"]["short_code"] == "SMU"
        assert result[1]["degree_normalized"]["canonical_name"] == "Accountancy"

    def test_bulk_unknown_institution(self, norm):
        """Unknown institution → institution_normalized is None."""
        records = [{"institution": "Hogwarts", "degree": "Computer Science"}]
        result = norm.bulk_normalize_education(records)
        assert result[0]["institution_normalized"] is None
        assert result[0]["degree_normalized"] is not None  # degree still resolves

    def test_bulk_empty_list(self, norm):
        assert norm.bulk_normalize_education([]) == []

    def test_bulk_missing_keys(self, norm):
        """Records without 'institution' or 'degree' should not crash."""
        records = [{"institution": "NTU"}, {"degree": "Accountancy"}, {}]
        result = norm.bulk_normalize_education(records)
        assert len(result) == 3
        for rec in result:
            assert "institution_normalized" in rec
            assert "degree_normalized" in rec


# ══════════════════════════════════════════════════════════════════
#  GRACEFUL DEGRADATION — missing DB
# ══════════════════════════════════════════════════════════════════

class TestMissingDatabase:

    def test_missing_db_does_not_raise(self, tmp_path):
        n = SGInstitutionNormalizer(tmp_path / "nonexistent.db")
        assert n.available is False

    def test_normalize_institution_returns_none_when_unavailable(self, tmp_path):
        n = SGInstitutionNormalizer(tmp_path / "nonexistent.db")
        assert n.normalize_institution("NTU") is None

    def test_normalize_degree_returns_none_when_unavailable(self, tmp_path):
        n = SGInstitutionNormalizer(tmp_path / "nonexistent.db")
        assert n.normalize_degree("Computer Science") is None

    def test_bulk_normalize_returns_list_when_unavailable(self, tmp_path):
        n = SGInstitutionNormalizer(tmp_path / "nonexistent.db")
        records = [{"institution": "NTU", "degree": "CS"}]
        result = n.bulk_normalize_education(records)
        assert len(result) == 1
        assert result[0]["institution_normalized"] is None
        assert result[0]["degree_normalized"] is None
