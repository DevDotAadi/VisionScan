"""Unit tests for VisionScan Global — Google Gemini client & prompts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from utils.prompt_templates import (
    get_explanation_prompt,
    get_pdf_narrative_prompt,
    get_chatbot_prompt,
    DISCLAIMER_TEXT,
)
from utils.gemini_client import (
    load_gemini_client,
    generate_explanation,
    generate_pdf_narrative,
    answer_user_question,
)


class TestPromptTemplates:
    def test_explanation_contains_disclaimer_and_inputs(self):
        prompt = get_explanation_prompt(
            prediction="Malignant",
            confidence=0.924,
            risk_level="High Risk",
            certainty="Certain",
            recommendation="Consult doctor.",
            language="English",
        )
        assert DISCLAIMER_TEXT in prompt
        assert "Malignant" in prompt
        assert "92.40%" in prompt
        assert "High Risk" in prompt

    def test_pdf_narrative_rules(self):
        prompt = get_pdf_narrative_prompt("Benign", 0.99, "Low Risk", "Certain", "Routine review.")
        assert DISCLAIMER_TEXT in prompt
        assert "exactly one highly professional paragraph" in prompt

    def test_chatbot_context_injection(self):
        prompt = get_chatbot_prompt(
            question="What should I do?",
            current_prediction={"label": "Malignant", "confidence": 0.88, "risk": "High Risk", "certainty": "Certain"},
        )
        assert DISCLAIMER_TEXT in prompt
        assert "Malignant" in prompt
        assert "88.0%" in prompt


class TestGeminiClient:
    @patch("utils.gemini_client.os.getenv")
    def test_load_client_none_if_missing_key(self, mock_getenv):
        mock_getenv.return_value = ""
        # Force clear cached client
        with patch("utils.gemini_client._GENAI_CLIENT", None):
            client = load_gemini_client()
            assert client is None

    @patch("utils.gemini_client.load_gemini_client")
    def test_fallback_called_on_api_error(self, mock_load):
        # Setup mock client that raises an exception when generate_content is called
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API error")
        mock_load.return_value = mock_client

        explanation = generate_explanation("Malignant", 0.92, "High Risk", "Certain", "Consult expert.")
        assert "Local ML Assessment Result" in explanation
        assert "Malignant" in explanation

    @patch("utils.gemini_client.load_gemini_client")
    def test_narrative_fallback_on_error(self, mock_load):
        mock_load.return_value = None  # Force fallback
        narrative = generate_pdf_narrative("Benign", 0.95, "Low Risk", "Certain", "Review monthly.")
        assert "Low Risk" in narrative
        assert DISCLAIMER_TEXT in narrative

    @patch("utils.gemini_client.load_gemini_client")
    def test_answer_question_fallback_on_error(self, mock_load):
        mock_load.return_value = None  # Force fallback
        ans = answer_user_question("Is this dangerous?")
        assert DISCLAIMER_TEXT in ans
