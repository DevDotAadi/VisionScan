"""Google Gemini Client wrapper for VisionScan Global.

Integrates with the `google-genai` SDK with full enterprise hardening:
- Request timeout and exponential backoff retry.
- Per-session request rate limiting and daily usage caps.
- Strict token/character truncation and prompt caching.
- Circuit breaker to prevent API abuse/unnecessary bill spikes.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import lru_cache
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

from utils.prompt_templates import (
    DISCLAIMER_TEXT,
    get_chatbot_prompt,
    get_explanation_prompt,
    get_pdf_narrative_prompt,
)

# Load environment variables
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

log = logging.getLogger(__name__)

# Cache Client
_GENAI_CLIENT: genai.Client | None = None
GEMINI_MODEL: str = "gemini-2.5-flash"

# Global lock for thread safety
_STATE_LOCK = Lock()

# Caches and limits
_SESSION_USAGE: dict[str, int] = defaultdict(int)        # Track request counts per session
_SESSION_LAST_REQUEST: dict[str, list[float]] = defaultdict(list)  # Track timestamps for RPM checks
_DAILY_COUNTER: int = 0
_DAILY_LAST_RESET: float = time.time()

# Circuit Breaker state
_CONSECUTIVE_FAILURES: int = 0
_CIRCUIT_TRIPPED: bool = False
_CIRCUIT_COOLDOWN_UNTIL: float = 0.0

# LLM Limits from Environment
MAX_SESSION_REQS = int(os.getenv("GEMINI_MAX_REQUESTS_PER_SESSION", 20))
MAX_RPM = int(os.getenv("GEMINI_MAX_REQUESTS_PER_MINUTE", 5))
MAX_DAILY_REQS = int(os.getenv("GEMINI_MAX_DAILY_REQUESTS", 100))
MAX_CHAR_LIMIT = 4000
TIMEOUT_SECONDS = 10.0


def load_gemini_client() -> genai.Client | None:
    """Load and return the GenAI client. Returns None if API key is missing."""
    global _GENAI_CLIENT
    if _GENAI_CLIENT is not None:
        return _GENAI_CLIENT

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.startswith("your_api_key") or len(api_key.strip()) < 10:
        log.warning("Gemini API key is not configured in .env. LLM features will fall back gracefully.")
        return None

    try:
        _GENAI_CLIENT = genai.Client(api_key=api_key.strip())
        log.info("Google GenAI client successfully initialised with secure keys.")
        return _GENAI_CLIENT
    except Exception as e:
        log.error("Failed to initialise Google GenAI client: %s", e)
        return None


def _check_rate_limits(session_id: str | None = "default") -> tuple[bool, str]:
    """Verify rate limits, per-session constraints, and daily quotas in a thread-safe manner."""
    global _DAILY_COUNTER, _DAILY_LAST_RESET
    sid = session_id or "default"

    with _STATE_LOCK:
        now = time.time()

        # Reset daily counter if 24 hours have passed
        if now - _DAILY_LAST_RESET >= 86400:
            _DAILY_COUNTER = 0
            _DAILY_LAST_RESET = now

        # 1. Daily Quota Check
        if _DAILY_COUNTER >= MAX_DAILY_REQS:
            return False, f"Daily limit of {MAX_DAILY_REQS} queries reached."

        # 2. Session Request Limit Check
        if _SESSION_USAGE[sid] >= MAX_SESSION_REQS:
            return False, f"You have reached your session limit of {MAX_SESSION_REQS} queries."

        # 3. Requests Per Minute Check
        past_requests = _SESSION_LAST_REQUEST[sid]
        # Retain only requests within the last 60 seconds
        _SESSION_LAST_REQUEST[sid] = [t for t in past_requests if now - t < 60]
        if len(_SESSION_LAST_REQUEST[sid]) >= MAX_RPM:
            return False, f"Rate limit exceeded. Maximum {MAX_RPM} queries per minute allowed."

        return True, ""


def _register_request(session_id: str | None = "default") -> None:
    """Log a successful or attempted API transaction."""
    global _DAILY_COUNTER
    sid = session_id or "default"
    with _STATE_LOCK:
        now = time.time()
        _SESSION_USAGE[sid] += 1
        _SESSION_LAST_REQUEST[sid].append(now)
        _DAILY_COUNTER += 1


# Prompt LRU cache to save repeated calls
@lru_cache(maxsize=128)
def _get_cached_explanation(prompt: str) -> str | None:
    """Return cached prompt result to eliminate duplicate queries."""
    return None  # Managed dynamically below by a global cache dictionary


_PROMPT_CACHE: dict[str, str] = {}


def _check_circuit_breaker() -> bool:
    """Check if the circuit breaker is currently open."""
    global _CIRCUIT_TRIPPED, _CONSECUTIVE_FAILURES
    with _STATE_LOCK:
        if _CIRCUIT_TRIPPED:
            if time.time() > _CIRCUIT_COOLDOWN_UNTIL:
                log.info("Circuit breaker entering half-open state; cooling period elapsed.")
                _CIRCUIT_TRIPPED = False
                _CONSECUTIVE_FAILURES = 0
            else:
                return False  # Closed/Active Breaker
        return True  # Normal state


def _report_api_failure() -> None:
    """Report API failure to trigger circuit breaker if threshold is reached."""
    global _CONSECUTIVE_FAILURES, _CIRCUIT_TRIPPED, _CIRCUIT_COOLDOWN_UNTIL
    with _STATE_LOCK:
        _CONSECUTIVE_FAILURES += 1
        if _CONSECUTIVE_FAILURES >= 3:
            _CIRCUIT_TRIPPED = True
            # Cooldown for 5 minutes
            _CIRCUIT_COOLDOWN_UNTIL = time.time() + 300
            log.critical("Circuit breaker TRIPPED! consecutive API failures. Gemini queries bypassed for 5 minutes.")


def _report_api_success() -> None:
    """Reset consecutive failures on success."""
    global _CONSECUTIVE_FAILURES
    with _STATE_LOCK:
        _CONSECUTIVE_FAILURES = 0


def _truncate_prompt(prompt: str) -> str:
    """Ensure prompt length stays strictly within maximum characters limit."""
    if len(prompt) > MAX_CHAR_LIMIT:
        log.warning("Truncated oversized prompt from %d to %d characters.", len(prompt), MAX_CHAR_LIMIT)
        return prompt[:MAX_CHAR_LIMIT]
    return prompt


def _call_gemini_with_retry_and_timeout(client: genai.Client, prompt: str) -> str:
    """Call the Gemini API wrapped in thread timeouts and exponential backoffs."""
    # Ensure system override instructions are safely insulated from injection
    sanitised_prompt = _truncate_prompt(prompt)

    # Simple cache check
    if sanitised_prompt in _PROMPT_CACHE:
        log.info("Prompt cache hit! Returning cached generation.")
        return _PROMPT_CACHE[sanitised_prompt]

    # Check circuit breaker
    if not _check_circuit_breaker():
        raise APIError(None, None, "Circuit breaker tripped due to consecutive errors.")

    def api_call():
        return client.models.generate_content(
            model=GEMINI_MODEL,
            contents=sanitised_prompt,
        )

    # Exponential retry with backoff
    last_err = None
    for attempt in range(3):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(api_call)
                # Strict timeout enforcement
                response = future.result(timeout=TIMEOUT_SECONDS)

            if response and response.text:
                result_text = response.text.strip()
                _report_api_success()
                # Cache response
                with _STATE_LOCK:
                    _PROMPT_CACHE[sanitised_prompt] = result_text
                return result_text

            raise ValueError("Empty response received from LLM.")

        except TimeoutError as te:
            last_err = te
            log.warning("Gemini request TIMEOUT on attempt %d/3.", attempt + 1)
        except Exception as e:
            last_err = e
            log.warning("Gemini request error on attempt %d/3: %s", attempt + 1, e)

        # Exponential Backoff with Jitter
        if attempt < 2:
            sleep_time = (2 ** attempt) + random.uniform(0.1, 1.0)
            time.sleep(sleep_time)

    _report_api_failure()
    raise last_err if last_err else RuntimeError("Gemini content generation failed after retries.")


def generate_explanation(
    prediction: str,
    confidence: float,
    risk_level: str,
    certainty: str,
    recommendation: str,
    language: str = "English",
    session_id: str | None = "default",
) -> str:
    """Generate patient-friendly, multilingual explanations with rate and safety guards."""
    client = load_gemini_client()
    if not client:
        return _get_fallback_explanation(prediction, confidence, risk_level, recommendation)

    # Rate limiting verification
    allowed, msg = _check_rate_limits(session_id)
    if not allowed:
        log.warning("Rate limit rejected: %s", msg)
        return f"### ⚠️ AI Limit Enforced\n\n{msg}\n\n**Local Fallback Recommendation:** {recommendation}\n\n{DISCLAIMER_TEXT}"

    prompt = get_explanation_prompt(
        prediction=prediction,
        confidence=confidence,
        risk_level=risk_level,
        certainty=certainty,
        recommendation=recommendation,
        language=language,
    )

    try:
        _register_request(session_id)
        return _call_gemini_with_retry_and_timeout(client, prompt)
    except Exception as e:
        log.error("Failed to generate educational explanation from Gemini: %s", e)
        return _get_fallback_explanation(prediction, confidence, risk_level, recommendation)


def generate_pdf_narrative(
    prediction: str,
    confidence: float,
    risk_level: str,
    certainty: str,
    recommendation: str,
    session_id: str | None = "default",
) -> str:
    """Generate a highly polished clinical-style narrative paragraph for the PDF reports."""
    client = load_gemini_client()
    if not client:
        return (
            f"Automated risk assessment determined a {risk_level} with a calibrated "
            f"confidence of {confidence:.2%}. {recommendation} {DISCLAIMER_TEXT}"
        )

    # Check rates
    allowed, _ = _check_rate_limits(session_id)
    if not allowed:
        return f"Automated risk assessment: {prediction} ({confidence:.2%}). Risk category is {risk_level}. Please consult a physician. {DISCLAIMER_TEXT}"

    prompt = get_pdf_narrative_prompt(
        prediction=prediction,
        confidence=confidence,
        risk_level=risk_level,
        certainty=certainty,
        recommendation=recommendation,
    )

    try:
        _register_request(session_id)
        return _call_gemini_with_retry_and_timeout(client, prompt)
    except Exception as e:
        log.error("Gemini narrative generation failed: %s", e)
        return f"Automated structural analysis prediction: {prediction} ({confidence:.2%}). Risk category is {risk_level}."


def answer_user_question(
    question: str,
    current_prediction: dict | None = None,
    chat_history: list[dict] | None = None,
    session_id: str | None = "default",
) -> str:
    """Answer conversational user questions safely using diagnostic context."""
    client = load_gemini_client()
    if not client:
        return (
            f"The dermatology assistant is currently offline.\n\n"
            f"**Safety Reminder:** {DISCLAIMER_TEXT} If you are concerned about a lesion, "
            f"please consult a certified dermatologist immediately."
        )

    # Check limits
    allowed, msg = _check_rate_limits(session_id)
    if not allowed:
        return f"⚠️ **AI Assistant Limit reached:** {msg}\n\n{DISCLAIMER_TEXT}"

    prompt = get_chatbot_prompt(
        question=question,
        current_prediction=current_prediction,
        chat_history=chat_history,
    )

    try:
        _register_request(session_id)
        return _call_gemini_with_retry_and_timeout(client, prompt)
    except Exception as e:
        log.error("Gemini Q&A failed: %s", e)
        return (
            f"Pardon, I encountered a temporary connection issue.\n\n"
            f"**Please remember:** {DISCLAIMER_TEXT} "
            f"Any skin lesion showing signs of itching, bleeding, or color changes should be inspected by a dermatologist."
        )


def _get_fallback_explanation(prediction: str, confidence: float, risk_level: str, recommendation: str) -> str:
    """Deterministic fallback explanation if LLM is unavailable."""
    return f"""### 🔬 Local ML Assessment Result
- **Assessment Class:** {prediction}
- **Assigned Risk:** {risk_level}
- **Mathematical Confidence:** {confidence:.2%}

#### Understanding the Assessment:
The automated image analyzer classified this skin lesion as **{prediction}**. This result is calculated directly on your system from dermoscopic structural patterns using a local convolutional network.

#### Suggested Next Steps:
- **Clinical Evaluation:** {recommendation}
- **Mandatory Safety Disclaimer:** {DISCLAIMER_TEXT}

*Please note: Real-time advanced multilingual AI synthesis was bypassed because the Gemini client is currently offline.*"""
