"""
gemini_client.py

💎✨ FAIRY CODEMOTHER'S SIMPLE GEMINI WRAPPER ✨💎

Lightweight REST-based Gemini client. No SDK install needed, just `requests`.

Usage:
    from gemini_client import GeminiClient
    
    gem = GeminiClient()  # reads GEMINI_API_KEY from environment
    response = gem.generate("Hello, how are you?")
    print(response)

Setup:
    1. Get API key: https://aistudio.google.com/app/apikey
    2. Add to .env: GEMINI_API_KEY=AIzaSy...
"""

import os
import time
import json
from typing import Optional, Dict, Any
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class GeminiClient:
    """Simple Gemini REST API client with retry + rate-limit handling."""

    # Free tier models (as of 2026)
    MODELS = {
        "flash-lite": "gemini-2.5-flash-lite",   # 15-30 RPM, 1000+ RPD (recommended for bulk)
        "flash":      "gemini-2.5-flash",        # 10-15 RPM, 250 RPD  (better quality)
        "pro":        "gemini-2.5-pro",          # 5 RPM, 50-100 RPD   (best quality, tight)
    }

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: Optional[str] = None, model: str = "flash-lite"):
        """
        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env.
            model: One of 'flash-lite', 'flash', 'pro' (or a raw model ID).
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "💔 No Gemini API key found!\n"
                "   Get one at: https://aistudio.google.com/app/apikey\n"
                "   Then add to .env: GEMINI_API_KEY=AIzaSy..."
            )

        # Resolve model name (allow shortcut or full name)
        self.model = self.MODELS.get(model, model)
        self.endpoint = f"{self.BASE_URL}/{self.model}:generateContent"

        # Rate limiting (conservative for free tier)
        self.min_interval_seconds = 4.5  # ~13 RPM safely under 15 RPM limit
        self.last_call_time = 0.0

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> Optional[str]:
        """
        Send a prompt to Gemini, return the text response.
        Handles rate limits with exponential backoff.

        Returns None if the call fails after all retries.
        """
        # Throttle to stay under free-tier RPM
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

        payload = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "topP": 0.9,
            }
        }

        url = f"{self.endpoint}?key={self.api_key}"

        for attempt in range(max_retries):
            try:
                self.last_call_time = time.time()
                response = requests.post(
                    url,
                    json=payload,
                    timeout=timeout,
                    headers={"Content-Type": "application/json"},
                )

                # 429 = rate limit hit (our fault — too fast)
                if response.status_code == 429:
                    wait = (2 ** attempt) * 5
                    print(f"   ⏳ Gemini rate limit (429), waiting {wait}s...")
                    time.sleep(wait)
                    continue

                # 503 = Gemini servers overloaded (their fault — wait longer)
                if response.status_code == 503:
                    wait = (2 ** attempt) * 15  # longer waits for server issues
                    print(f"   🌧️  Gemini overloaded (503), waiting {wait}s...")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()

                # Parse the response
                candidates = data.get("candidates", [])
                if not candidates:
                    print(f"   ⚠️  Gemini returned no candidates")
                    return None

                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if not parts:
                    print(f"   ⚠️  Gemini returned empty content")
                    return None

                text = "".join(p.get("text", "") for p in parts)
                return text.strip() if text else None

            except requests.exceptions.Timeout:
                print(f"   ⏱️  Gemini timeout (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except requests.exceptions.HTTPError as e:
                status = response.status_code if 'response' in locals() else 0
                # 5xx errors retry, 4xx errors give up
                if status >= 500:
                    wait = (2 ** attempt) * 10
                    print(f"   🌧️  Gemini server error {status}, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"   ❌ Gemini HTTP {status}: {e}")
                    return None
            except Exception as e:
                print(f"   ❌ Gemini error: {e}")
                return None

        # All retries exhausted
        print(f"   💔 Gemini gave up after {max_retries} retries")
        return None

    def test_connection(self) -> bool:
        """Quick test to verify API key and connectivity."""
        result = self.generate("Reply with just the word 'OK'.", max_output_tokens=10)
        if result and "OK" in result.upper():
            print(f"✅ Gemini {self.model} connection OK")
            return True
        print(f"❌ Gemini connection failed (got: {result!r})")
        return False


# ─── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🧪 Testing Gemini connection...")
    client = GeminiClient(model="flash-lite")
    if client.test_connection():
        print(f"\n💎 Using model: {client.model}")
        print("✨ Ready to integrate into stage0_lite_generate.py!")
    else:
        print("\n💔 Setup incomplete. Check your GEMINI_API_KEY in .env")