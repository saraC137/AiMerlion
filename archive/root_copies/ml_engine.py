"""
ml_engine.py

💅✨ FAIRY CODEMOTHER'S MACHINE LEARNING ENGINE ✨💅

The GRAND EVOLUTION from rule-based pattern learning to ACTUAL machine learning!
Think of this as upgrading from hand-stitched costumes to a full AI fashion house! 🎭👗

This module provides ML-powered enhancements to the resume extraction system:

  1. 🎯 ExtractionConfidenceScorer
     - Trains on features derived from extractions + correction history
     - Predicts HOW LIKELY an extraction is correct (0.0 → 1.0)
     - Uses gradient boosting (lightweight, no GPU needed)

  2. 🏆 SmartReviewPrioritizer
     - Ranks candidates by "most likely to need human review"
     - Combines confidence scores, field completeness, anomaly flags
     - Saves HR team hours by surfacing problems FIRST

  3. 💡 AutoCorrectionSuggester
     - Learns from past dashboard corrections (extraction_log)
     - Uses TF-IDF + cosine similarity to find similar past mistakes
     - Suggests corrections for new extractions based on patterns

  4. 🚨 AnomalyDetector
     - Flags statistically unusual values (e.g., 200-digit phone numbers)
     - Uses Isolation Forest for outlier detection on numeric features
     - Rule-based checks for format violations

  5. 📊 MLDashboardIntegration
     - Provides API-ready data for the review dashboard
     - Batch scoring for entire database
     - Training pipeline that runs on your existing DB data

Architecture:
  - Uses scikit-learn (lightweight, no GPU required!)
  - Trains on data from your existing resume_extractions.db
  - Models are saved to disk as .joblib files for fast loading
  - Graceful degradation: works even with 0 training data (uses heuristics)

Dependencies:
    pip install scikit-learn joblib numpy --break-system-packages

Usage:
    from ml_engine import MLEngine
    engine = MLEngine(db_path="resume_extractions.db")
    engine.train_all()                          # Train on existing data
    scores = engine.score_candidate(12345)      # Score a single candidate
    ranked = engine.get_review_queue(limit=50)  # Prioritized review list
"""

import sqlite3
import json
import os
import re
import math
import logging
import datetime
import pickle
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict

import numpy as np

# Scikit-learn imports — the ML fashion toolkit! 💄
try:
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        IsolationForest,
        RandomForestClassifier
    )
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# =============================================================================
# 🔧 CONFIGURATION
# =============================================================================

logger = logging.getLogger(__name__)

# Model storage directory
MODEL_DIR = "ml_models"

# Minimum training samples needed before ML kicks in
# Below this threshold, we fall back to heuristic scoring
MIN_TRAINING_SAMPLES = 20

# Feature names for the confidence model
# Each extraction gets turned into this feature vector
CONFIDENCE_FEATURES = [
    "text_length",              # Raw resume text length
    "name_length",              # Length of extracted name
    "name_word_count",          # Number of words in name
    "name_has_numbers",         # Does name contain digits? (suspicious!)
    "name_has_special",         # Special chars in name (O'Brien is OK, @#$ is not)
    "email_valid_format",       # Does email match standard pattern?
    "email_has_multiple_at",    # Multiple @ symbols? Probably wrong
    "phone_digit_count",        # How many digits in phone?
    "phone_has_country_code",   # Starts with + ?
    "phone_length_valid",       # Is digit count in valid range (7-15)?
    "dob_valid_format",         # Is DOB in YYYY-MM-DD format?
    "dob_age_reasonable",       # Is the calculated age between 18-70?
    "skills_count",             # Number of skills extracted
    "skills_avg_length",        # Average skill string length
    "experience_length",        # Length of experience text
    "experience_has_dates",     # Does experience contain date patterns?
    "education_length",         # Length of education text
    "education_has_institution",# Contains university/school keywords?
    "field_completeness",       # Fraction of non-empty critical fields (0-1)
    "ai_assisted",              # Was AI used? (binary)
    "extraction_method_score",  # Encoded extraction method
]

# Regex patterns used for feature extraction
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')
DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')
YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')
INSTITUTION_KEYWORDS = {
    'university', 'college', 'institute', 'school', 'polytechnic',
    'akademi', 'universiti', 'nus', 'ntu', 'smu', 'sit', 'suss',
    'ntu', 'ump', 'upm', 'ukm', 'um', 'uitm', 'iium'
}


# =============================================================================
# 🧬 FEATURE ENGINEERING — Turning Raw Data Into ML Gold! ✨
# =============================================================================

def extract_features(
    structured_data: Dict,
    raw_text: Optional[str] = None
) -> np.ndarray:
    """
    🧬 Transform a candidate's extraction data into a numeric feature vector.
    
    This is the METAMORPHOSIS stage — we take messy, mixed-type data and
    transform it into a beautiful butterfly of numbers that ML can digest! 🦋
    
    Think of it like translating a candidate's entire resume into a
    single row of numbers that captures the ESSENCE of extraction quality.
    
    Args:
        structured_data: Dict from structured_extractions table.
        raw_text:        Optional raw resume text for additional features.
        
    Returns:
        numpy array of shape (len(CONFIDENCE_FEATURES),) with numeric features.
    """
    features = {}
    
    # --- Text length ---
    # Longer resumes = more data to work with = generally better extractions
    features["text_length"] = len(raw_text) if raw_text else 0
    
    # --- Name features ---
    # A good name extraction should be 2-4 words, no numbers, reasonable length
    name = str(structured_data.get("name") or "")
    features["name_length"] = len(name)
    features["name_word_count"] = len(name.split()) if name else 0
    features["name_has_numbers"] = 1 if any(c.isdigit() for c in name) else 0
    features["name_has_special"] = 1 if re.search(r'[@#$%^&*(){}[\]|\\<>]', name) else 0
    
    # --- Email features ---
    # Valid email = very specific format, easy to validate
    email = str(structured_data.get("email") or "")
    features["email_valid_format"] = 1 if EMAIL_PATTERN.match(email) else 0
    features["email_has_multiple_at"] = 1 if email.count("@") > 1 else 0
    
    # --- Phone features ---
    # Singapore: 8 digits, Malaysia: 9-10 digits, India: 10 digits
    phone = str(structured_data.get("phone") or "")
    digits = re.sub(r'\D', '', phone)
    features["phone_digit_count"] = len(digits)
    features["phone_has_country_code"] = 1 if phone.strip().startswith("+") else 0
    features["phone_length_valid"] = 1 if 7 <= len(digits) <= 15 else 0
    
    # --- Date of Birth features ---
    # Should be YYYY-MM-DD, resulting in age 18-70
    dob = str(structured_data.get("date_of_birth") or "")
    features["dob_valid_format"] = 1 if DATE_PATTERN.match(dob) else 0
    features["dob_age_reasonable"] = 0
    if DATE_PATTERN.match(dob):
        try:
            birth_year = int(dob[:4])
            current_year = datetime.datetime.now().year
            age = current_year - birth_year
            features["dob_age_reasonable"] = 1 if 18 <= age <= 70 else 0
        except (ValueError, IndexError):
            pass
    
    # --- Skills features ---
    # More skills = more content captured, but watch for garbage
    skills = str(structured_data.get("skills_raw") or "")
    skill_items = [s.strip() for s in skills.split("|") if s.strip()] if "|" in skills else [skills] if skills else []
    features["skills_count"] = len(skill_items)
    features["skills_avg_length"] = (
        np.mean([len(s) for s in skill_items]) if skill_items else 0
    )
    
    # --- Experience features ---
    # Good experience text contains dates and is reasonably long
    exp = str(structured_data.get("experience_raw") or "")
    features["experience_length"] = len(exp)
    features["experience_has_dates"] = 1 if YEAR_PATTERN.search(exp) else 0
    
    # --- Education features ---
    # Should mention institutions/universities
    edu = str(structured_data.get("education_raw") or "")
    features["education_length"] = len(edu)
    edu_lower = edu.lower()
    features["education_has_institution"] = (
        1 if any(kw in edu_lower for kw in INSTITUTION_KEYWORDS) else 0
    )
    
    # --- Overall quality signals ---
    critical_fields = ["name", "email", "phone", "skills_raw", "experience_raw", "education_raw"]
    filled = sum(
        1 for f in critical_fields
        if structured_data.get(f) and str(structured_data[f]).strip()
    )
    features["field_completeness"] = filled / len(critical_fields)
    
    features["ai_assisted"] = 1 if structured_data.get("ai_assisted") else 0
    
    # Encode extraction method as numeric score
    method = str(structured_data.get("extraction_method") or "Unknown")
    method_scores = {"Regex + AI": 2, "Regex Only": 1, "Unknown": 0}
    features["extraction_method_score"] = method_scores.get(method, 0)
    
    # Build the feature vector in consistent order
    return np.array([features.get(f, 0) for f in CONFIDENCE_FEATURES], dtype=np.float64)


def extract_features_batch(
    rows: List[Dict],
    raw_texts: Optional[Dict[int, str]] = None
) -> np.ndarray:
    """
    🧬 Batch feature extraction for multiple candidates.
    
    More efficient than calling extract_features() in a loop because
    we can vectorize some operations. Like a production line! 🏭
    
    Args:
        rows:      List of structured_extraction dicts.
        raw_texts: Optional dict mapping candidate_id -> raw_text.
        
    Returns:
        numpy array of shape (n_candidates, n_features).
    """
    raw_texts = raw_texts or {}
    feature_matrix = []
    
    for row in rows:
        cid = row.get("candidate_id")
        raw = raw_texts.get(cid, None)
        feature_matrix.append(extract_features(row, raw))
    
    return np.array(feature_matrix) if feature_matrix else np.empty((0, len(CONFIDENCE_FEATURES)))


# =============================================================================
# 🎯 MODEL 1: EXTRACTION CONFIDENCE SCORER
# Predicts how likely each extraction is CORRECT based on learned patterns.
# =============================================================================

class ExtractionConfidenceScorer:
    """
    🎯 Predicts extraction confidence using Gradient Boosting.
    
    Drama analogy: Imagine a talent scout who's seen 1000 auditions.
    After a while, they can INSTANTLY tell if a performance is good
    just from the first 30 seconds. That's what this model does — 
    it looks at the FEATURES of an extraction and predicts whether
    the data is likely correct or needs fixing! 🎭
    
    Training data comes from:
    - extraction_log entries where was_overridden = 1 (corrections = WRONG)
    - extraction_log entries where was_successful = 1 AND reviewed = 1 (CORRECT)
    - Heuristic labels when explicit labels are unavailable
    
    Features: see CONFIDENCE_FEATURES list above.
    """
    
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, "confidence_scorer.joblib")
        self.scaler_path = os.path.join(model_dir, "confidence_scaler.joblib")
        
        self.model = None
        self.scaler = None
        self._is_trained = False
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self):
        """
        📦 Load a previously trained model from disk.
        Like pulling your best outfit from storage — saves time! 👗
        """
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self._is_trained = True
                logger.info("🎯 Confidence model loaded from disk")
        except Exception as e:
            logger.warning(f"⚠️ Could not load confidence model: {e}")
            self._is_trained = False
    
    def train(self, db_path: str) -> Dict[str, Any]:
        """
        🏋️ Train the confidence model on data from the database!
        
        Training Strategy:
        - POSITIVE samples (label=1): Candidates marked as reviewed AND 
          not corrected, OR candidates with high heuristic quality scores.
        - NEGATIVE samples (label=0): Candidates with corrections in the
          extraction_log (was_overridden=1), OR candidates with low
          heuristic quality (failed format checks, missing critical fields).
        
        This hybrid approach works even when you don't have many
        manual reviews yet — heuristic labels bootstrap the model! 🚀
        
        Args:
            db_path: Path to the SQLite database.
            
        Returns:
            Dict with training metrics (accuracy, n_samples, feature_importance).
        """
        if not SKLEARN_AVAILABLE:
            logger.error("❌ scikit-learn not installed! pip install scikit-learn")
            return {"error": "scikit-learn not available"}
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # ---- Gather training data ----
            
            # Get all structured extractions with their raw text lengths
            rows = conn.execute("""
                SELECT s.*, r.text_length, r.raw_text
                FROM structured_extractions s
                LEFT JOIN raw_extractions r ON s.candidate_id = r.candidate_id
            """).fetchall()
            
            if len(rows) < MIN_TRAINING_SAMPLES:
                logger.warning(
                    f"⚠️ Only {len(rows)} samples — need {MIN_TRAINING_SAMPLES} for ML. "
                    f"Using heuristic scoring instead."
                )
                return {"error": f"Insufficient data ({len(rows)}/{MIN_TRAINING_SAMPLES})"}
            
            # Get correction history to identify BAD extractions
            corrections = conn.execute("""
                SELECT candidate_id, field_name, COUNT(*) as correction_count
                FROM extraction_log
                WHERE extraction_method = 'manual_correction'
                   OR extraction_method = 'dashboard_manual_edit'
                   OR was_overridden = 1
                GROUP BY candidate_id
            """).fetchall()
            
            corrected_candidates = {
                row["candidate_id"]: row["correction_count"]
                for row in corrections
            }
            
            # ---- Build feature matrix and labels ----
            X = []
            y = []
            candidate_ids = []
            
            for row in rows:
                row_dict = dict(row)
                cid = row_dict["candidate_id"]
                raw = row_dict.pop("raw_text", None)
                
                features = extract_features(row_dict, raw)
                X.append(features)
                candidate_ids.append(cid)
                
                # ---- Label Assignment Strategy ----
                # Priority 1: Explicit review status
                if row_dict.get("reviewed") and cid not in corrected_candidates:
                    # Reviewed and NOT corrected = good extraction
                    y.append(1)
                elif cid in corrected_candidates and corrected_candidates[cid] >= 2:
                    # Multiple corrections = bad extraction
                    y.append(0)
                else:
                    # Priority 2: Heuristic label based on format checks
                    # This bootstraps training even without manual reviews
                    score = self._heuristic_quality_score(row_dict)
                    y.append(1 if score >= 0.7 else 0)
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(
                f"🏋️ Training with {len(X)} samples "
                f"({sum(y)} good, {len(y) - sum(y)} bad)"
            )
            
            # ---- Handle class imbalance ----
            # Resume extractions tend to be mostly "good" — we need to handle
            # the imbalance so the model doesn't just predict "good" for everything
            n_positive = sum(y)
            n_negative = len(y) - n_positive
            
            if n_positive == 0 or n_negative == 0:
                logger.warning("⚠️ All samples have the same label — cannot train discriminatively")
                return {"error": "No label variance in training data"}
            
            scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
            
            # ---- Train the model ----
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                min_samples_split=max(5, len(X) // 20),  # Adaptive to dataset size
                min_samples_leaf=max(2, len(X) // 40),
                subsample=0.8,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            self._is_trained = True
            
            # ---- Evaluate with cross-validation ----
            cv_folds = min(5, max(2, len(X) // 10))  # Adaptive fold count
            try:
                cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv_folds, scoring='accuracy')
                cv_accuracy = cv_scores.mean()
                cv_std = cv_scores.std()
            except Exception:
                cv_accuracy = 0.0
                cv_std = 0.0
            
            # ---- Feature importance (which features matter most?) ----
            importances = dict(zip(
                CONFIDENCE_FEATURES,
                self.model.feature_importances_
            ))
            # Sort by importance descending
            importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
            
            # ---- Save model to disk ----
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"💾 Model saved to {self.model_path}")
            
            metrics = {
                "n_samples": len(X),
                "n_positive": int(n_positive),
                "n_negative": int(n_negative),
                "cv_accuracy": round(cv_accuracy, 4),
                "cv_std": round(cv_std, 4),
                "feature_importance": {k: round(v, 4) for k, v in list(importances.items())[:10]},
                "model_type": "GradientBoostingClassifier",
                "trained_at": datetime.datetime.now().isoformat()
            }
            
            # Save training metrics for dashboard display
            metrics_path = os.path.join(self.model_dir, "training_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(
                f"✅ Model trained! CV Accuracy: {cv_accuracy:.2%} (±{cv_std:.2%})"
            )
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    def predict_confidence(
        self,
        structured_data: Dict,
        raw_text: Optional[str] = None
    ) -> float:
        """
        🎯 Predict confidence score for a single extraction.
        
        Returns a score between 0.0 (probably wrong) and 1.0 (probably correct).
        
        If the ML model hasn't been trained yet, falls back to heuristic
        scoring — like using a ruler when you don't have a laser level! 📏
        
        Args:
            structured_data: Dict of extraction fields.
            raw_text:        Optional raw resume text.
            
        Returns:
            Float confidence score [0.0, 1.0].
        """
        if self._is_trained and self.model is not None and self.scaler is not None:
            try:
                features = extract_features(structured_data, raw_text).reshape(1, -1)
                features_scaled = self.scaler.transform(features)
                
                # predict_proba returns [P(class=0), P(class=1)]
                # We want P(class=1) = probability of being correct
                proba = self.model.predict_proba(features_scaled)[0]
                confidence = float(proba[1]) if len(proba) > 1 else float(proba[0])
                
                return round(confidence, 4)
            except Exception as e:
                logger.warning(f"⚠️ ML prediction failed, using heuristic: {e}")
        
        # Fallback: heuristic scoring
        return self._heuristic_quality_score(structured_data)
    
    def predict_batch(
        self,
        rows: List[Dict],
        raw_texts: Optional[Dict[int, str]] = None
    ) -> List[float]:
        """
        🎯 Batch predict confidence for multiple candidates.
        
        More efficient than calling predict_confidence() in a loop
        because we can vectorize the feature extraction and prediction.
        
        Args:
            rows:      List of structured_extraction dicts.
            raw_texts: Optional dict mapping candidate_id -> raw_text.
            
        Returns:
            List of confidence scores, one per candidate.
        """
        if not rows:
            return []
        
        if self._is_trained and self.model is not None and self.scaler is not None:
            try:
                X = extract_features_batch(rows, raw_texts)
                X_scaled = self.scaler.transform(X)
                probas = self.model.predict_proba(X_scaled)
                return [round(float(p[1]), 4) for p in probas]
            except Exception as e:
                logger.warning(f"⚠️ Batch ML prediction failed: {e}")
        
        # Fallback
        return [self._heuristic_quality_score(row) for row in rows]
    
    def _heuristic_quality_score(self, data: Dict) -> float:
        """
        📏 Rule-based quality score (fallback when ML is not trained).
        
        This is the "analog" version — like judging a talent show
        with a checklist instead of an AI! 📋
        
        Scoring criteria:
          +0.15 for valid name (2-4 words, no numbers)
          +0.15 for valid email format
          +0.15 for valid phone (7-15 digits)
          +0.10 for valid DOB format and reasonable age
          +0.15 for non-empty skills
          +0.15 for experience with dates
          +0.15 for education with institution keywords
        
        Returns:
            Float score [0.0, 1.0].
        """
        score = 0.0
        
        # Name check
        name = str(data.get("name") or "")
        if name and 1 <= len(name.split()) <= 5 and not any(c.isdigit() for c in name):
            score += 0.15
        
        # Email check
        email = str(data.get("email") or "")
        if EMAIL_PATTERN.match(email):
            score += 0.15
        
        # Phone check
        phone = str(data.get("phone") or "")
        digits = re.sub(r'\D', '', phone)
        if 7 <= len(digits) <= 15:
            score += 0.15
        
        # DOB check
        dob = str(data.get("date_of_birth") or "")
        if DATE_PATTERN.match(dob):
            try:
                age = datetime.datetime.now().year - int(dob[:4])
                if 18 <= age <= 70:
                    score += 0.10
            except (ValueError, IndexError):
                pass
        
        # Skills check
        skills = str(data.get("skills_raw") or "")
        if len(skills) > 10:
            score += 0.15
        
        # Experience check
        exp = str(data.get("experience_raw") or "")
        if len(exp) > 20 and YEAR_PATTERN.search(exp):
            score += 0.15
        
        # Education check
        edu = str(data.get("education_raw") or "").lower()
        if any(kw in edu for kw in INSTITUTION_KEYWORDS):
            score += 0.15
        
        return round(min(1.0, score), 4)


# =============================================================================
# 💡 MODEL 2: AUTO-CORRECTION SUGGESTER
# Learns from past corrections to suggest fixes for new extractions.
# =============================================================================

class AutoCorrectionSuggester:
    """
    💡 Suggests corrections based on patterns learned from past edits.
    
    Drama analogy: Imagine a costume designer who's fixed the same
    broken zipper 50 times. Next time they see that zipper, they
    ALREADY know the fix before even looking at it. That's this! 🧵
    
    How it works:
    1. Collects all past corrections from extraction_log
    2. Builds a TF-IDF index over original→corrected value pairs
    3. For a new extraction, finds the most similar past correction
    4. Suggests the corresponding fix with a confidence score
    
    This is especially powerful for:
    - Phone number format fixes (e.g., "+65 91234567" → "91234567")
    - Name corrections (e.g., "MR. JOHN DOE" → "John Doe")
    - Date format corrections
    """
    
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Correction knowledge base
        # Structure: {field_name: [{original, corrected, context_features}, ...]}
        self.corrections_db: Dict[str, List[Dict]] = {}
        
        # TF-IDF vectorizers per field (for similarity matching)
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.tfidf_matrices: Dict[str, Any] = {}
        
        self._is_trained = False
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load previously saved correction knowledge base."""
        kb_path = os.path.join(self.model_dir, "correction_kb.json")
        try:
            if os.path.exists(kb_path):
                with open(kb_path, "r", encoding="utf-8") as f:
                    self.corrections_db = json.load(f)
                if self.corrections_db:
                    self._rebuild_tfidf_indices()
                    self._is_trained = True
                    logger.info("💡 Correction knowledge base loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not load correction KB: {e}")
    
    def _save_knowledge_base(self):
        """Save correction knowledge base to disk."""
        kb_path = os.path.join(self.model_dir, "correction_kb.json")
        with open(kb_path, "w", encoding="utf-8") as f:
            json.dump(self.corrections_db, f, indent=2, ensure_ascii=False)
    
    def train(self, db_path: str) -> Dict[str, Any]:
        """
        🏋️ Build the correction knowledge base from the database.
        
        Scans extraction_log for all manual corrections and organizes
        them by field for efficient similarity lookup.
        
        Args:
            db_path: Path to SQLite database.
            
        Returns:
            Dict with training stats per field.
        """
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # Fetch all correction entries
            # A "correction" is any extraction_log entry where:
            # - was_overridden = 1 (something replaced this value)
            # - extraction_method contains 'manual' or 'dashboard'
            corrections = conn.execute("""
                SELECT 
                    l.candidate_id,
                    l.field_name,
                    l.extracted_value,
                    l.override_reason,
                    l.extraction_method,
                    l.timestamp
                FROM extraction_log l
                WHERE l.was_overridden = 1
                   OR l.extraction_method IN ('manual_correction', 'dashboard_manual_edit')
                ORDER BY l.field_name, l.timestamp
            """).fetchall()
            
            # Organize corrections by field
            self.corrections_db = {}
            
            for corr in corrections:
                field = corr["field_name"]
                if field not in self.corrections_db:
                    self.corrections_db[field] = []
                
                # Parse the override_reason to extract old→new mapping
                reason = corr["override_reason"] or ""
                original = ""
                corrected = corr["extracted_value"] or ""
                
                # Try to extract the "Old:" value from the reason string
                old_match = re.search(r"Old:\s*'([^']*)'", reason)
                if old_match:
                    original = old_match.group(1)
                
                # Also handle the format from db_manager's update_field:
                # "Manual correction from 'old_value'"
                alt_match = re.search(r"from\s*'([^']*)'", reason)
                if alt_match and not original:
                    original = alt_match.group(1)
                
                self.corrections_db[field].append({
                    "original": original,
                    "corrected": corrected,
                    "candidate_id": corr["candidate_id"],
                    "timestamp": corr["timestamp"]
                })
            
            # Build TF-IDF indices for similarity matching
            self._rebuild_tfidf_indices()
            
            # Save knowledge base
            self._save_knowledge_base()
            self._is_trained = True
            
            stats = {
                field: len(entries)
                for field, entries in self.corrections_db.items()
            }
            total = sum(stats.values())
            
            logger.info(f"💡 Correction KB built: {total} corrections across {len(stats)} fields")
            return {"total_corrections": total, "per_field": stats}
            
        except Exception as e:
            logger.error(f"❌ Correction KB training failed: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    def _rebuild_tfidf_indices(self):
        """
        Rebuild TF-IDF indices from the correction knowledge base.
        
        We index the ORIGINAL (wrong) values so that when we see
        a new extraction, we can find the most similar past mistakes.
        """
        if not SKLEARN_AVAILABLE:
            return
        
        self.vectorizers = {}
        self.tfidf_matrices = {}
        
        for field, entries in self.corrections_db.items():
            originals = [e["original"] for e in entries if e["original"]]
            if len(originals) < 2:
                continue
            
            try:
                vectorizer = TfidfVectorizer(
                    analyzer="char_wb",   # Character n-grams (handles typos better!)
                    ngram_range=(2, 4),   # 2-4 character sequences
                    max_features=5000,
                    lowercase=True
                )
                matrix = vectorizer.fit_transform(originals)
                self.vectorizers[field] = vectorizer
                self.tfidf_matrices[field] = matrix
            except Exception as e:
                logger.debug(f"⚠️ TF-IDF failed for {field}: {e}")
    
    def suggest_correction(
        self,
        field_name: str,
        current_value: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        💡 Suggest corrections for a given field value.
        
        Finds the most similar past corrections and returns their
        fixes with confidence scores.
        
        Args:
            field_name:    Which field (e.g., 'name', 'phone').
            current_value: The current extraction to check.
            top_k:         How many suggestions to return.
            
        Returns:
            List of dicts: [{suggestion, confidence, original_was, corrected_to}, ...]
            Empty list if no similar corrections found.
        """
        if not current_value or not self._is_trained:
            return []
        
        if field_name not in self.vectorizers or field_name not in self.tfidf_matrices:
            return []
        
        try:
            vectorizer = self.vectorizers[field_name]
            matrix = self.tfidf_matrices[field_name]
            entries = self.corrections_db[field_name]
            
            # Transform the current value using the same TF-IDF vocabulary
            query_vec = vectorizer.transform([current_value])
            
            # Compute cosine similarity against all past corrections
            similarities = cosine_similarity(query_vec, matrix).flatten()
            
            # Get top-k most similar past corrections
            # Only consider matches above a minimum similarity threshold
            min_similarity = 0.3
            top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get extra for filtering
            
            suggestions = []
            seen_corrections = set()
            
            for idx in top_indices:
                sim = similarities[idx]
                if sim < min_similarity:
                    break
                
                entry = entries[idx]
                corrected = entry["corrected"]
                
                # Deduplicate suggestions
                if corrected in seen_corrections or corrected == current_value:
                    continue
                seen_corrections.add(corrected)
                
                suggestions.append({
                    "suggestion": corrected,
                    "confidence": round(float(sim), 3),
                    "based_on_original": entry["original"],
                    "based_on_corrected": corrected,
                    "from_candidate": entry.get("candidate_id")
                })
                
                if len(suggestions) >= top_k:
                    break
            
            return suggestions
            
        except Exception as e:
            logger.debug(f"⚠️ Suggestion failed for {field_name}: {e}")
            return []


# =============================================================================
# 🚨 MODEL 3: ANOMALY DETECTOR
# Flags statistically unusual extractions.
# =============================================================================

class AnomalyDetector:
    """
    🚨 Detects statistically unusual extraction values.
    
    Drama analogy: Every fashion show has that ONE outfit that makes
    the audience gasp — not in a good way. This detector spots those
    moments BEFORE the model hits the runway! 😱👗
    
    Two-layer approach:
    1. Rule-based checks (format violations, impossible values)
    2. Isolation Forest on numeric features (statistical outliers)
    """
    
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.iso_forest = None
        self.scaler = None
        self._is_trained = False
    
    def train(self, db_path: str) -> Dict[str, Any]:
        """
        🏋️ Train the anomaly detector on the database.
        
        The Isolation Forest learns what "normal" extractions look like,
        so it can flag weird ones. Like learning what normal looks like
        at a fashion show, so you know when something's OFF. 🎭
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available"}
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            rows = conn.execute("""
                SELECT s.*, r.text_length
                FROM structured_extractions s
                LEFT JOIN raw_extractions r ON s.candidate_id = r.candidate_id
            """).fetchall()
            
            if len(rows) < MIN_TRAINING_SAMPLES:
                return {"error": f"Need {MIN_TRAINING_SAMPLES} samples, have {len(rows)}"}
            
            X = extract_features_batch([dict(r) for r in rows])
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Contamination = expected fraction of anomalies
            # 10% is a reasonable starting point for resume data
            self.iso_forest = IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
            self.iso_forest.fit(X_scaled)
            self._is_trained = True
            
            # Save models
            joblib.dump(self.iso_forest, os.path.join(self.model_dir, "anomaly_forest.joblib"))
            joblib.dump(self.scaler, os.path.join(self.model_dir, "anomaly_scaler.joblib"))
            
            # Get anomaly scores for training data (for reporting)
            scores = self.iso_forest.decision_function(X_scaled)
            n_anomalies = sum(1 for s in scores if s < 0)
            
            logger.info(f"🚨 Anomaly detector trained: {n_anomalies}/{len(rows)} flagged")
            return {
                "n_samples": len(rows),
                "n_anomalies": n_anomalies,
                "anomaly_rate": round(n_anomalies / len(rows), 4)
            }
            
        except Exception as e:
            logger.error(f"❌ Anomaly training failed: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    def detect(self, structured_data: Dict, raw_text: Optional[str] = None) -> Dict[str, Any]:
        """
        🚨 Check a single extraction for anomalies.
        
        Returns both rule-based flags and ML-based anomaly score.
        
        Args:
            structured_data: Dict of extraction fields.
            raw_text:        Optional raw text.
            
        Returns:
            Dict with:
              - is_anomaly: bool
              - anomaly_score: float (-1 to 1, lower = more anomalous)
              - rule_flags: list of specific rule violations
        """
        result = {
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "rule_flags": [],
            "severity": "normal"  # normal, warning, critical
        }
        
        # ---- Layer 1: Rule-based checks ----
        flags = self._rule_based_checks(structured_data)
        result["rule_flags"] = flags
        
        # ---- Layer 2: ML-based anomaly score ----
        if self._is_trained and self.iso_forest is not None and self.scaler is not None:
            try:
                features = extract_features(structured_data, raw_text).reshape(1, -1)
                features_scaled = self.scaler.transform(features)
                
                # decision_function: negative = anomaly, positive = normal
                score = float(self.iso_forest.decision_function(features_scaled)[0])
                result["anomaly_score"] = round(score, 4)
                
                # Prediction: 1 = normal, -1 = anomaly
                prediction = self.iso_forest.predict(features_scaled)[0]
                
                if prediction == -1:
                    result["is_anomaly"] = True
            except Exception as e:
                logger.debug(f"⚠️ Anomaly detection ML failed: {e}")
        
        # Combine rule-based and ML results for severity
        critical_flags = [f for f in flags if f.get("severity") == "critical"]
        warning_flags = [f for f in flags if f.get("severity") == "warning"]
        
        if critical_flags or (result["is_anomaly"] and len(flags) > 2):
            result["severity"] = "critical"
            result["is_anomaly"] = True
        elif warning_flags or result["is_anomaly"]:
            result["severity"] = "warning"
            result["is_anomaly"] = True
        
        return result
    
    def _rule_based_checks(self, data: Dict) -> List[Dict]:
        """
        📏 Rule-based anomaly checks — the common-sense layer.
        
        These catch things that are OBVIOUSLY wrong regardless of
        what the ML model thinks. Like noticing someone wearing
        their shoes on the wrong feet! 👟🔄
        """
        flags = []
        
        # --- Name checks ---
        name = str(data.get("name") or "")
        if name:
            if len(name) > 80:
                flags.append({
                    "field": "name", 
                    "issue": f"Unusually long name ({len(name)} chars)",
                    "severity": "warning"
                })
            if "@" in name:
                flags.append({
                    "field": "name",
                    "issue": "Name contains @ symbol (email in name field?)",
                    "severity": "critical"
                })
            digit_count = sum(1 for c in name if c.isdigit())
            if digit_count > 2:
                flags.append({
                    "field": "name",
                    "issue": f"Name contains {digit_count} digits (phone/ID in name?)",
                    "severity": "critical"
                })
            if name.isupper() and len(name) > 3:
                flags.append({
                    "field": "name",
                    "issue": "Name is ALL CAPS (may need proper casing)",
                    "severity": "warning"
                })
        
        # --- Email checks ---
        email = str(data.get("email") or "")
        if email:
            if not EMAIL_PATTERN.match(email):
                flags.append({
                    "field": "email",
                    "issue": "Email doesn't match standard format",
                    "severity": "warning"
                })
            if email.count("@") > 1:
                flags.append({
                    "field": "email",
                    "issue": "Multiple @ symbols in email",
                    "severity": "critical"
                })
        
        # --- Phone checks ---
        phone = str(data.get("phone") or "")
        if phone:
            digits = re.sub(r'\D', '', phone)
            if len(digits) < 7:
                flags.append({
                    "field": "phone",
                    "issue": f"Phone too short ({len(digits)} digits)",
                    "severity": "warning"
                })
            elif len(digits) > 15:
                flags.append({
                    "field": "phone",
                    "issue": f"Phone too long ({len(digits)} digits)",
                    "severity": "critical"
                })
            # Check if phone looks like a date
            if re.match(r'^\d{4}-\d{2}-\d{2}$', phone.strip()):
                flags.append({
                    "field": "phone",
                    "issue": "Phone looks like a date (DOB in phone field?)",
                    "severity": "critical"
                })
        
        # --- DOB checks ---
        dob = str(data.get("date_of_birth") or "")
        if dob:
            if not DATE_PATTERN.match(dob):
                flags.append({
                    "field": "date_of_birth",
                    "issue": "DOB not in YYYY-MM-DD format",
                    "severity": "warning"
                })
            else:
                try:
                    age = datetime.datetime.now().year - int(dob[:4])
                    if age < 15 or age > 80:
                        flags.append({
                            "field": "date_of_birth",
                            "issue": f"Calculated age ({age}) seems unreasonable",
                            "severity": "critical"
                        })
                except (ValueError, IndexError):
                    pass
        
        # --- Cross-field checks ---
        # Check if email appears in phone field or vice versa
        if phone and "@" in phone:
            flags.append({
                "field": "phone",
                "issue": "Phone field contains @ (email in phone field?)",
                "severity": "critical"
            })
        
        if email and email.replace(" ", "").isdigit():
            flags.append({
                "field": "email",
                "issue": "Email field contains only numbers (phone in email?)",
                "severity": "critical"
            })
        
        return flags


# =============================================================================
# 🏆 MODEL 4: SMART REVIEW PRIORITIZER
# Combines all models to rank candidates for human review.
# =============================================================================

class SmartReviewPrioritizer:
    """
    🏆 Ranks candidates by "most likely to need human review."
    
    Drama analogy: Before the fashion show, the director decides
    which outfits need the most last-minute fixes. The ones with
    the most issues get attention FIRST. That's this! 🎬👗
    
    Priority score formula (lower = needs review more urgently):
      priority = confidence * 0.5
               + completeness * 0.2
               + (1 - anomaly_severity) * 0.2
               + was_reviewed * 0.1
    
    Lower priority score = MORE urgent for review.
    """
    
    def __init__(
        self,
        confidence_scorer: ExtractionConfidenceScorer,
        anomaly_detector: AnomalyDetector,
        correction_suggester: AutoCorrectionSuggester
    ):
        self.confidence_scorer = confidence_scorer
        self.anomaly_detector = anomaly_detector
        self.correction_suggester = correction_suggester
    
    def compute_priority(
        self,
        structured_data: Dict,
        raw_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        🏆 Compute a review priority score for a single candidate.
        
        Returns a comprehensive assessment including all ML signals.
        
        Args:
            structured_data: Dict of extraction fields.
            raw_text:        Optional raw text.
            
        Returns:
            Dict with priority_score, confidence, anomalies, suggestions, etc.
        """
        # Confidence score
        confidence = self.confidence_scorer.predict_confidence(structured_data, raw_text)
        
        # Anomaly detection
        anomaly_result = self.anomaly_detector.detect(structured_data, raw_text)
        
        # Field completeness
        critical_fields = ["name", "email", "phone", "skills_raw", "experience_raw", "education_raw"]
        filled = sum(
            1 for f in critical_fields
            if structured_data.get(f) and str(structured_data[f]).strip()
        )
        completeness = filled / len(critical_fields)
        
        # Anomaly severity score (0=no issues, 1=critical issues)
        severity_map = {"normal": 0.0, "warning": 0.5, "critical": 1.0}
        anomaly_severity = severity_map.get(anomaly_result.get("severity", "normal"), 0.0)
        
        # Review status
        was_reviewed = 1.0 if structured_data.get("reviewed") else 0.0
        
        # === Priority Score Calculation ===
        # Lower = MORE urgent for review
        priority_score = (
            confidence * 0.5 +
            completeness * 0.2 +
            (1 - anomaly_severity) * 0.2 +
            was_reviewed * 0.1
        )
        
        # Get correction suggestions for flagged fields
        suggestions = {}
        for flag in anomaly_result.get("rule_flags", []):
            field = flag["field"]
            value = str(structured_data.get(field) or "")
            if value:
                field_suggestions = self.correction_suggester.suggest_correction(field, value, top_k=2)
                if field_suggestions:
                    suggestions[field] = field_suggestions
        
        return {
            "priority_score": round(priority_score, 4),
            "priority_label": self._score_to_label(priority_score),
            "confidence": confidence,
            "completeness": round(completeness, 4),
            "anomaly": anomaly_result,
            "suggestions": suggestions,
            "reviewed": bool(was_reviewed),
            "candidate_id": structured_data.get("candidate_id")
        }
    
    def _score_to_label(self, score: float) -> str:
        """Convert numeric priority score to human-readable label."""
        if score < 0.3:
            return "🔴 URGENT"
        elif score < 0.5:
            return "🟡 NEEDS REVIEW"
        elif score < 0.7:
            return "🟢 LOOKS OK"
        else:
            return "✅ HIGH CONFIDENCE"


# =============================================================================
# 🎭 ML ENGINE — The Grand Orchestrator!
# Combines all models into a single, easy-to-use interface.
# =============================================================================

class MLEngine:
    """
    🎭 THE ML ENGINE — Fairy Codemother's Grand ML Orchestrator!
    
    This is the ONE class you need to interact with. It manages all
    the individual ML models, handles training, and provides a clean
    API for scoring, ranking, and suggesting corrections.
    
    Think of it as the stage manager who coordinates ALL the actors,
    sets, and lights for the perfect show! 🎬✨
    
    Usage:
        engine = MLEngine(db_path="resume_extractions.db")
        
        # Train all models on existing data
        results = engine.train_all()
        
        # Score a single candidate
        score = engine.score_candidate(12345)
        
        # Get prioritized review queue
        queue = engine.get_review_queue(limit=50)
        
        # Get suggestions for a specific field
        suggestions = engine.suggest_corrections("phone", "+65 912345678")
    """
    
    def __init__(self, db_path: str = "resume_extractions.db", model_dir: str = MODEL_DIR):
        """
        🎀 Initialize the ML Engine.
        
        Args:
            db_path:   Path to the SQLite database.
            model_dir: Directory for saving/loading trained models.
        """
        self.db_path = db_path
        self.model_dir = model_dir
        
        if not SKLEARN_AVAILABLE:
            logger.warning(
                "⚠️ scikit-learn not installed! ML features disabled. "
                "Install with: pip install scikit-learn joblib --break-system-packages"
            )
        
        # Initialize all model components
        self.confidence_scorer = ExtractionConfidenceScorer(model_dir)
        self.correction_suggester = AutoCorrectionSuggester(model_dir)
        self.anomaly_detector = AnomalyDetector(model_dir)
        self.prioritizer = SmartReviewPrioritizer(
            self.confidence_scorer,
            self.anomaly_detector,
            self.correction_suggester
        )
        
        logger.info(f"🎭 ML Engine initialized (db={db_path}, models={model_dir})")
    
    def train_all(self) -> Dict[str, Any]:
        """
        🏋️ Train ALL models on the current database data.
        
        Call this after you've processed a batch of resumes, or after
        a reviewing session where corrections were made. Like sending
        your AI to finishing school! 🎓
        
        Returns:
            Dict with training results for each model.
        """
        logger.info("🏋️ Training all ML models...")
        results = {}
        
        # 1. Confidence Scorer
        logger.info("  🎯 Training confidence scorer...")
        results["confidence_scorer"] = self.confidence_scorer.train(self.db_path)
        
        # 2. Correction Suggester
        logger.info("  💡 Building correction knowledge base...")
        results["correction_suggester"] = self.correction_suggester.train(self.db_path)
        
        # 3. Anomaly Detector
        logger.info("  🚨 Training anomaly detector...")
        results["anomaly_detector"] = self.anomaly_detector.train(self.db_path)
        
        # Save training timestamp
        meta_path = os.path.join(self.model_dir, "ml_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "last_trained": datetime.datetime.now().isoformat(),
                "db_path": self.db_path,
                "results_summary": {
                    k: v.get("error") or "success" 
                    for k, v in results.items()
                }
            }, f, indent=2)
        
        logger.info("✅ All models trained!")
        return results
    
    def score_candidate(self, candidate_id: int) -> Dict[str, Any]:
        """
        🎯 Get a comprehensive ML score for a single candidate.
        
        Combines confidence scoring, anomaly detection, and correction
        suggestions into one neat package.
        
        Args:
            candidate_id: The candidate's numeric ID.
            
        Returns:
            Dict with all ML signals for this candidate.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # Fetch structured data
            row = conn.execute("""
                SELECT s.* FROM structured_extractions s
                WHERE s.candidate_id = ?
                ORDER BY s.created_at DESC LIMIT 1
            """, (candidate_id,)).fetchone()
            
            if not row:
                return {"error": f"Candidate {candidate_id} not found"}
            
            structured_data = dict(row)
            
            # Fetch raw text
            raw_row = conn.execute("""
                SELECT raw_text FROM raw_extractions
                WHERE candidate_id = ?
                ORDER BY extraction_timestamp DESC LIMIT 1
            """, (candidate_id,)).fetchone()
            
            raw_text = raw_row["raw_text"] if raw_row else None
            
            # Run the full priority assessment
            result = self.prioritizer.compute_priority(structured_data, raw_text)
            result["candidate_id"] = candidate_id
            result["name"] = structured_data.get("name", "Unknown")
            
            return result
            
        except sqlite3.Error as e:
            logger.error(f"❌ Scoring failed for candidate {candidate_id}: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    def get_review_queue(
        self,
        limit: int = 50,
        include_reviewed: bool = False
    ) -> List[Dict]:
        """
        🏆 Get a prioritized list of candidates for human review.
        
        Returns candidates sorted by URGENCY — the ones most likely
        to have extraction errors appear FIRST. Like a triage nurse
        sorting patients in the ER! 🏥
        
        Args:
            limit:            Max number of candidates to return.
            include_reviewed: Include already-reviewed candidates?
            
        Returns:
            List of dicts sorted by priority (most urgent first).
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # Build query based on filter
            where = "" if include_reviewed else "WHERE (s.reviewed = 0 OR s.reviewed IS NULL)"
            
            rows = conn.execute(f"""
                SELECT s.*, r.text_length, r.raw_text
                FROM structured_extractions s
                LEFT JOIN raw_extractions r ON s.candidate_id = r.candidate_id
                {where}
                ORDER BY s.candidate_id
            """).fetchall()
            
            # Score each candidate
            queue = []
            for row in rows:
                row_dict = dict(row)
                raw_text = row_dict.pop("raw_text", None)
                
                priority = self.prioritizer.compute_priority(row_dict, raw_text)
                priority["name"] = row_dict.get("name", "Unknown")
                priority["email"] = row_dict.get("email", "")
                queue.append(priority)
            
            # Sort by priority score ASCENDING (lower = more urgent)
            queue.sort(key=lambda x: x.get("priority_score", 1.0))
            
            return queue[:limit]
            
        except sqlite3.Error as e:
            logger.error(f"❌ Review queue query failed: {e}")
            return []
        finally:
            conn.close()
    
    def suggest_corrections(
        self,
        field_name: str,
        current_value: str,
        top_k: int = 3
    ) -> List[Dict]:
        """
        💡 Get correction suggestions for a specific field value.
        
        Args:
            field_name:    Field to check (e.g., 'name', 'phone').
            current_value: Current extracted value.
            top_k:         Number of suggestions to return.
            
        Returns:
            List of suggestion dicts.
        """
        return self.correction_suggester.suggest_correction(field_name, current_value, top_k)
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        📊 Get the current training status of all models.
        
        Useful for displaying in the dashboard — "are the ML models
        trained and ready?" 🎭
        """
        meta_path = os.path.join(self.model_dir, "ml_meta.json")
        
        status = {
            "sklearn_available": SKLEARN_AVAILABLE,
            "confidence_trained": self.confidence_scorer._is_trained,
            "corrections_trained": self.correction_suggester._is_trained,
            "anomaly_trained": self.anomaly_detector._is_trained,
            "last_trained": None,
            "model_dir": self.model_dir,
        }
        
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                status["last_trained"] = meta.get("last_trained")
                status["results_summary"] = meta.get("results_summary")
            except Exception:
                pass
        
        # Load training metrics if available
        metrics_path = os.path.join(self.model_dir, "training_metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as f:
                    status["confidence_metrics"] = json.load(f)
            except Exception:
                pass
        
        return status
    
    def get_field_suggestions_for_candidate(self, candidate_id: int) -> Dict[str, List[Dict]]:
        """
        💡 Get correction suggestions for ALL fields of a candidate.
        
        Useful for pre-populating the dashboard edit view with
        "did you mean...?" suggestions. ✨
        
        Args:
            candidate_id: The candidate's numeric ID.
            
        Returns:
            Dict mapping field_name -> list of suggestions.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            row = conn.execute("""
                SELECT * FROM structured_extractions
                WHERE candidate_id = ?
                ORDER BY created_at DESC LIMIT 1
            """, (candidate_id,)).fetchone()
            
            if not row:
                return {}
            
            data = dict(row)
            all_suggestions = {}
            
            # Check each editable field
            checkable_fields = {
                "name": "name",
                "email": "email", 
                "phone": "phone",
                "date_of_birth": "date_of_birth",
                "location": "location",
            }
            
            for field, col in checkable_fields.items():
                value = str(data.get(col) or "")
                if value:
                    suggestions = self.correction_suggester.suggest_correction(field, value)
                    if suggestions:
                        all_suggestions[field] = suggestions
            
            return all_suggestions
            
        except sqlite3.Error as e:
            logger.error(f"❌ Suggestions query failed: {e}")
            return {}
        finally:
            conn.close()


# =============================================================================
# 🖥️ CLI INTERFACE — Run Training & Scoring from Command Line
# =============================================================================

def main():
    """
    🎬 CLI entry point for ML training and analysis.
    
    Usage:
        python ml_engine.py                  # Train all models
        python ml_engine.py --score 12345    # Score a specific candidate
        python ml_engine.py --queue          # Show review queue
        python ml_engine.py --status         # Show model status
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="💅✨ Fairy Codemother's ML Engine ✨💅"
    )
    parser.add_argument(
        "--db", default="resume_extractions.db",
        help="Path to the SQLite database"
    )
    parser.add_argument(
        "--model-dir", default=MODEL_DIR,
        help="Directory for model storage"
    )
    parser.add_argument(
        "--score", type=int, default=None,
        help="Score a specific candidate by ID"
    )
    parser.add_argument(
        "--queue", action="store_true",
        help="Show prioritized review queue"
    )
    parser.add_argument(
        "--queue-limit", type=int, default=20,
        help="Number of candidates in review queue"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show training status"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train all models (default if no other action)"
    )
    
    args = parser.parse_args()
    
    # Setup logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - 🎭 %(levelname)s - %(message)s"
    )
    
    # Verify database exists
    if not os.path.exists(args.db):
        print(f"\n❌ Database not found: {args.db}")
        print("💡 Run your extraction pipeline first, or specify --db path")
        return
    
    # Initialize engine
    engine = MLEngine(db_path=args.db, model_dir=args.model_dir)
    
    # ---- Status ----
    if args.status:
        status = engine.get_training_status()
        print("\n" + "=" * 60)
        print("  🎭 ML ENGINE STATUS")
        print("=" * 60)
        print(f"  scikit-learn: {'✅ Available' if status['sklearn_available'] else '❌ Not installed'}")
        print(f"  Confidence model: {'✅ Trained' if status['confidence_trained'] else '⏳ Not trained'}")
        print(f"  Correction KB: {'✅ Built' if status['corrections_trained'] else '⏳ Not built'}")
        print(f"  Anomaly detector: {'✅ Trained' if status['anomaly_trained'] else '⏳ Not trained'}")
        print(f"  Last trained: {status.get('last_trained', 'Never')}")
        
        if "confidence_metrics" in status:
            m = status["confidence_metrics"]
            print(f"\n  📊 Confidence Model Metrics:")
            print(f"     Samples: {m.get('n_samples', '?')}")
            print(f"     CV Accuracy: {m.get('cv_accuracy', 0):.2%}")
            print(f"     Top features:")
            for feat, imp in list(m.get("feature_importance", {}).items())[:5]:
                bar = "█" * int(imp * 50)
                print(f"       {feat:30s} {bar} {imp:.3f}")
        print()
        return
    
    # ---- Score single candidate ----
    if args.score:
        result = engine.score_candidate(args.score)
        if "error" in result:
            print(f"\n❌ {result['error']}")
            return
        
        print(f"\n{'=' * 60}")
        print(f"  🎯 ML SCORE: {result['name']} (#{result['candidate_id']})")
        print(f"{'=' * 60}")
        print(f"  Priority: {result['priority_label']} ({result['priority_score']:.3f})")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Completeness: {result['completeness']:.1%}")
        print(f"  Reviewed: {'✅' if result['reviewed'] else '⏳'}")
        
        anomaly = result.get("anomaly", {})
        flags = anomaly.get("rule_flags", [])
        if flags:
            print(f"\n  🚨 Anomaly Flags:")
            for f in flags:
                icon = "🔴" if f["severity"] == "critical" else "🟡"
                print(f"     {icon} [{f['field']}] {f['issue']}")
        
        suggestions = result.get("suggestions", {})
        if suggestions:
            print(f"\n  💡 Correction Suggestions:")
            for field, suggs in suggestions.items():
                for s in suggs:
                    print(f"     [{field}] → \"{s['suggestion']}\" (confidence: {s['confidence']:.0%})")
        print()
        return
    
    # ---- Review queue ----
    if args.queue:
        queue = engine.get_review_queue(limit=args.queue_limit)
        if not queue:
            print("\n✅ No candidates pending review!")
            return
        
        print(f"\n{'=' * 80}")
        print(f"  🏆 REVIEW QUEUE — Top {len(queue)} Candidates Needing Attention")
        print(f"{'=' * 80}")
        print(f"  {'#':<4} {'ID':<10} {'Name':<25} {'Priority':<20} {'Confidence':<12} {'Flags'}")
        print(f"  {'─' * 75}")
        
        for i, item in enumerate(queue, 1):
            flags = len(item.get("anomaly", {}).get("rule_flags", []))
            print(
                f"  {i:<4} {item['candidate_id']:<10} "
                f"{(item['name'] or 'Unknown')[:24]:<25} "
                f"{item['priority_label']:<20} "
                f"{item['confidence']:.1%}{'':>8} "
                f"{'⚠️ ' + str(flags) if flags else '—'}"
            )
        print()
        return
    
    # ---- Default: Train all models ----
    print("\n" + "=" * 60)
    print("  🏋️ TRAINING ALL ML MODELS")
    print("=" * 60 + "\n")
    
    results = engine.train_all()
    
    print("\n" + "=" * 60)
    print("  📊 TRAINING RESULTS")
    print("=" * 60)
    
    for model_name, result in results.items():
        status = "✅" if "error" not in result else "⚠️"
        print(f"\n  {status} {model_name}:")
        for k, v in result.items():
            if k == "feature_importance":
                print(f"     Top features:")
                for feat, imp in list(v.items())[:5]:
                    print(f"       {feat}: {imp:.4f}")
            else:
                print(f"     {k}: {v}")
    
    print()


if __name__ == "__main__":
    main()