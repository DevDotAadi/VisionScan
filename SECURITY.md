# Security Policy (SECURITY.md)

VisionScan Global is committed to creating a secure and responsible educational dermatology AI platform. This document outlines our threat model, security posture, and the process for reporting potential security issues.

---

## Supported Versions

We actively provide security patches for the following versions:

| Version | Supported |
| :--- | :--- |
| v1.0.x (Current) | Yes ✅ |
| < v1.0.0 | No ❌ |

---

## 🔒 Threat Model & Hardening Guardrails

To protect against unexpected charges, privacy leaks, and system abuse, we implement strict mitigation layers:

1. **Secret Isolation**: All API keys reside strictly in a localized `.env` file (excluded from Git). No secrets are logged or hardcoded.
2. **LLM Protection**: We wrap Gemini queries with fixed, hardened system instructions, strict prompt delimiters, input truncation, and a safety circuit breaker to prevent prompt injection and billing peaks.
3. **Streamlit Input Hardening**:
   - File uploads are validated via file extensions, MIME headers, Pillow image decode validation, and 10MB size limits.
   - All EXIF metadata is stripped from images on upload to preserve patient privacy.
   - Path traversal is blocked using `pathlib.Path` sandboxing.
   - All text inputs are HTML-escaped before display to eliminate XSS risks.
4. **Local Resilience**: In case of network drops, rate limits, or API-key absence, the application remains fully functional using local MobileNetV2 inference and deterministic local guidelines.

---

## 🛑 Reporting a Vulnerability

If you discover a security vulnerability, please do **NOT** open a public issue. Doing so risks exposing the issue before a fix is available.

Instead, please report vulnerabilities via email:
* **Contact**: security@visionscanglobal.org
* **Format**: Please include a detailed description of the vulnerability, steps to reproduce, a proof of concept (if available), and the potential impact.

We will acknowledge receipt of your report within **24 hours** and aim to provide a resolved patch within **7 business days**.

---

## 💡 Secret Handling Guidance

* **Never commit API keys**: Ensure your `.env` is listed inside `.gitignore` before making any commits.
* **Pre-Commit Checks**: We encourage using our configured `.pre-commit-config.yaml` to block accidental commits containing credential strings.
