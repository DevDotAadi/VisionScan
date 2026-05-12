"""VisionScan — Streamlit Web Application.

An educational AI platform for skin lesion risk assessment featuring
image upload, live camera, batch testing, explainability, PDF reports,
and Google Gemini 2.5 interactive integration.
Fully production-hardened, secure, and styled to modern Meta-inspired benchmarks.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from batch_test import run as run_batch
from gradcam import compute_heatmap, find_last_conv_layer, overlay
from pdf_report import generate as generate_pdf
from utils.db_utils import (
    get_prediction_history_df,
    init_db,
    log_prediction_to_db,
)
from utils.engine import (
    Prediction,
    RiskLevel,
    load_model,
    predict,
    preprocess,
)
from utils.gemini_client import (
    answer_user_question,
    generate_explanation,
    generate_pdf_narrative,
    load_gemini_client,
)
from utils.security_utils import (
    sanitize_text,
    secure_filename,
    strip_metadata,
    validate_image_upload,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Production Config and Limits ─────────────────────────────────────

MODEL_PATH = Path("models/visionscan_mobilenet.h5")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# Safety Configuration Defaults (Override via environment variables)
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", 10.0))
MAX_BATCH_IMAGES = int(os.getenv("MAX_BATCH_IMAGES", 500))

# Thread pool for non-blocking asynchronous multi-user inference
_INFERENCE_POOL = ThreadPoolExecutor(max_workers=4)

DISCLAIMER = (
    "This application is an educational AI tool and is not a substitute "
    "for professional medical diagnosis. Any concerning skin lesion should "
    "be evaluated by a qualified dermatologist."
)

# Premium Meta Theme Colors Matching Specifications
RISK_COLOURS: dict[str, str] = {
    "Low Risk": "#31A24C",         # Success Green
    "Moderate Risk": "#F7B928",    # Warning Yellow
    "High Risk": "#E41E3F",        # Danger Red
    "Inconclusive": "#A121CE",     # Oculus Purple / Inconclusive Accent
}

LANGUAGES = ["English", "Hindi", "Spanish", "French", "Arabic"]

# ── Page Config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="VisionScan",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Inlined Premium CSS (Guarantees Instant Hot-Reload & No Caching Lag) ──

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* Base Canvas Overrides - Use Poppins/Inter as defaults without breaking system icons */
html, body, .stApp {
    font-family: 'Poppins', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-color: #ffffff !important;
    color: #050505 !important;
    letter-spacing: -0.16px !important;
}

/* Explicitly restore Material icon fonts for any icon containers to solve the collapse button issue completely */
span[class*="material-symbols"],
span[class*="MaterialSymbols"],
[data-testid="stSidebarCollapseButton"] button span,
[data-testid="collapsedSidebar"] button span,
.material-icons,
[class*="material-icons"] {
    font-family: "Material Symbols Outlined", "Material Symbols Rounded", "Material Icons" !important;
}

/* Tighten Container Spacing */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1100px !important;
}

/* Tighten up Element Gap Spacings */
.stElementContainer {
    margin-bottom: 0.5rem !important;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.75rem !important;
}

/* Custom Headers Spacing */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    color: #050505 !important;
    letter-spacing: -0.02em !important;
    margin-top: 0.75rem !important;
    margin-bottom: 0.25rem !important;
}

h1 {
    font-size: 2.2rem !important;
    line-height: 1.1 !important;
}

h2 {
    font-size: 1.6rem !important;
}

h3 {
    font-size: 1.2rem !important;
}

/* Force button wrapper blocks to span 100% of container width to guarantee identical horizontal button lengths */
.stButton,
div.stDownloadButton,
[data-testid="stFileUploader"],
div[data-testid="stElementContainer"] div.stButton,
div[data-testid="stElementContainer"] div.stDownloadButton {
    width: 100% !important;
    min-width: 100% !important;
    display: block !important;
}

/* Unified Button Styling (Identical Size, Font, and Saturated Visual Strength) - Bigger, Longer, Stronger */
.stButton>button,
div.stDownloadButton > button,
[data-testid="stFileUploader"] button {
    width: 100% !important;
    min-width: 100% !important;
    height: 52px !important;
    border-radius: 100px !important;
    font-family: 'Poppins', 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 0 32px !important;
    background-color: #1877F2 !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 2px 4px rgba(24, 119, 242, 0.1) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Ensure child layers inside any button type inherit the high-contrast white value */
.stButton>button div, .stButton>button span, .stButton>button p,
div.stDownloadButton > button div, div.stDownloadButton > button span, div.stDownloadButton > button p,
[data-testid="stFileUploader"] button div, [data-testid="stFileUploader"] button span, [data-testid="stFileUploader"] button p {
    background-color: transparent !important;
    color: #ffffff !important;
    font-family: 'Poppins', 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}

.stButton>button:hover,
div.stDownloadButton > button:hover,
[data-testid="stFileUploader"] button:hover {
    background-color: #166FE5 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(24, 119, 242, 0.25) !important;
    border: none !important;
}

.stButton>button:hover div, .stButton>button:hover span, .stButton>button:hover p,
div.stDownloadButton > button:hover div, div.stDownloadButton > button:hover span, div.stDownloadButton > button:hover p,
[data-testid="stFileUploader"] button:hover div, [data-testid="stFileUploader"] button:hover span, [data-testid="stFileUploader"] button:hover p {
    color: #ffffff !important;
}

.stButton>button:active,
div.stDownloadButton > button:active,
[data-testid="stFileUploader"] button:active {
    background-color: #1464CC !important;
}

/* Premium Rounded Cards - Tightened */
.card {
    background-color: #ffffff !important;
    border: 1px solid #DADDE1 !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    text-align: left !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin-bottom: 12px !important;
}

.card:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.04) !important;
    border-color: #ccd0d5 !important;
}

.card h3 {
    margin: 0 0 4px 0 !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #050505 !important;
}

.card p {
    margin: 0 !important;
    color: #65676B !important;
    font-size: 0.9rem !important;
    line-height: 1.4 !important;
}

/* Statistical Indicator Cards */
.stat {
    background: #ffffff !important;
    border: 1px solid #DADDE1 !important;
    border-radius: 12px !important;
    padding: 14px !important;
    text-align: center !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.01) !important;
    transition: transform 0.2s ease !important;
}

.stat:hover {
    transform: translateY(-1px) !important;
    border-color: #1877F2 !important;
}

.stat h4 {
    margin: 0 !important;
    color: #65676B !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

.stat .v {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #1877F2 !important;
    margin-top: 4px !important;
}

/* Meta-Style Structured Banner Alerts - Tightened */
.banner {
    padding: 14px 20px !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    text-align: left !important;
    margin: 12px 0 !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
}

.banner h2 {
    margin: 0 !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    color: #ffffff !important;
}

.banner p {
    margin: 2px 0 0 0 !important;
    opacity: 0.95 !important;
    font-size: 0.9rem !important;
    color: #ffffff !important;
}

/* Sidebar Canvas and Separation - Tightened */
[data-testid="stSidebar"] {
    background-color: #F0F2F5 !important;
    border-right: 1px solid #DADDE1 !important;
}

[data-testid="stSidebar"] h2 {
    font-family: 'Poppins', sans-serif !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: #000000 !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.4rem !important;
}

/* Resolve Sidebar visibility conflicts and ensure 100% legibility of radio options */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [role="radiogroup"] label {
    background-color: transparent !important;
    padding: 4px 10px !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background-color: rgba(0, 0, 0, 0.03) !important;
}

/* Proactively force sidebar texts, headings, and labels to render with high-contrast ink-black values */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label div,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #1c1e21 !important;
    font-weight: 500 !important;
    font-family: 'Poppins', 'Inter', sans-serif !important;
}

/* Sidebar caption visibility */
[data-testid="stSidebar"] caption,
[data-testid="stSidebar"] .st-emotion-cache-1gulk7 {
    color: #65676B !important;
}

/* Make the sidebar collapse/expand floating button extremely visible, high contrast, and gorgeous */
[data-testid="stSidebarCollapseButton"] button,
button[data-testid="sidebar-collapse-button"] {
    background-color: #ffffff !important;
    border: 1px solid #DADDE1 !important;
    color: #1877F2 !important;
    border-radius: 50% !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    width: 32px !important;
    height: 32px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    z-index: 999999 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Force the expand button to float as a fixed circular button in the top left, completely bypassing any overflow or clipping boundaries */
[data-testid="collapsedSidebar"] button,
button[data-testid="sidebar-expand-button"],
div[data-testid="collapsedSidebar"] button {
    position: fixed !important;
    top: 8px !important;
    left: 8px !important;
    background-color: #ffffff !important;
    border: 1px solid #DADDE1 !important;
    color: #1877F2 !important;
    border-radius: 50% !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    width: 32px !important;
    height: 32px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    z-index: 9999999 !important;
    margin: 0 !important;
    padding: 0 !important;
}

[data-testid="stSidebarCollapseButton"] button:hover,
button[data-testid="sidebar-collapse-button"]:hover,
[data-testid="collapsedSidebar"] button:hover,
button[data-testid="sidebar-expand-button"]:hover,
div[data-testid="collapsedSidebar"] button:hover {
    background-color: #1877F2 !important;
    color: #ffffff !important;
    transform: scale(1.08) !important;
    box-shadow: 0 4px 12px rgba(24, 119, 242, 0.2) !important;
    border-color: #1877F2 !important;
}

[data-testid="stSidebarCollapseButton"] button svg,
[data-testid="stSidebarCollapseButton"] button span,
button[data-testid="sidebar-collapse-button"] svg,
button[data-testid="sidebar-collapse-button"] span,
[data-testid="collapsedSidebar"] button svg,
[data-testid="collapsedSidebar"] button span,
button[data-testid="sidebar-expand-button"] svg,
button[data-testid="sidebar-expand-button"] span,
div[data-testid="collapsedSidebar"] button svg,
div[data-testid="collapsedSidebar"] button span {
    color: inherit !important;
    fill: currentColor !important;
}

/* Input Fields and Selectors - Force high contrast readable black values on text selections */
div[data-baseweb="select"] > div, input, textarea {
    border-radius: 8px !important;
    border-color: #DADDE1 !important;
    color: #050505 !important;
    background-color: #ffffff !important;
    transition: all 0.2s ease !important;
}

div[data-baseweb="select"] span, div[data-baseweb="select"] div {
    color: #050505 !important;
}

div[data-baseweb="select"] > div:focus-within, input:focus, textarea:focus {
    border-color: #1877F2 !important;
    box-shadow: 0 0 0 2px rgba(24, 119, 242, 0.12) !important;
}

/* Ensure global label visibility for select-box labels, sliders, and form legends */
label, .stWidgetLabel, [data-testid="stWidgetLabel"] p {
    color: #050505 !important;
    font-weight: 500 !important;
}

/* Progress Bar Customisations */
div[role="progressbar"] {
    border-radius: 100px !important;
    overflow: hidden !important;
}

div[role="progressbar"] > div {
    background-color: #1877F2 !important;
}

/* Chat Bubbles (AI Assistant Page) */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 16px;
}

.chat-bubble {
    max-width: 75%;
    padding: 10px 14px;
    border-radius: 16px;
    font-size: 0.9rem;
    line-height: 1.4;
}

.chat-bubble.user {
    align-self: flex-end;
    background-color: #1877F2;
    color: #ffffff !important;
    border-bottom-right-radius: 4px;
}

.chat-bubble.user * {
    color: #ffffff !important;
}

.chat-bubble.assistant {
    align-self: flex-start;
    background-color: #F0F2F5;
    color: #050505 !important;
    border-bottom-left-radius: 4px;
    border: 1px solid #DADDE1;
}

.chat-bubble.assistant * {
    color: #050505 !important;
}

/* Styled Educational Recommendation Alerts */
.recommendation-alert {
    background-color: rgba(24, 119, 242, 0.05) !important;
    border-left: 4px solid #1877F2 !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
    margin: 12px 0 !important;
    color: #050505 !important;
}

/* Responsive Footer */
.footer {
    font-size: 0.8rem;
    color: #65676B;
    text-align: center;
    margin-top: 60px;
    padding: 16px 0;
    border-top: 1px solid #DADDE1;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────

@st.cache_resource
def _load():
    return load_model(MODEL_PATH)


def run_async_inference(model: tf.keras.Model, batch) -> Prediction:
    """Execute deep learning model prediction on a separate thread to prevent main UI blocking."""
    future = _INFERENCE_POOL.submit(predict, model, batch)
    return future.result()


def _log(filename: str, pred: Prediction) -> None:
    # Sanitise logs and save to concurrent sqlite database
    safe_name = secure_filename(filename)
    log_prediction_to_db(safe_name, pred)


def _hex(risk: RiskLevel) -> str:
    return RISK_COLOURS.get(risk.value, "#65676B")


# ── Pages ────────────────────────────────────────────────────────────

def page_home() -> None:
    # ── 1. Hero Section (Two-Column Layout) ──────────────────────────
    col_hero_left, col_hero_right = st.columns([6, 5], gap="large")

    with col_hero_left:
        st.markdown("""
        <div class="hero-badge">🌍 Open Source • Privacy First • Educational AI</div>
        <h1 class="hero-title">VisionScan</h1>
        <h2 class="hero-subtitle">AI-Powered Educational Skin Lesion Risk Assessment</h2>
        <p class="hero-desc">
            VisionScan utilizes advanced deep learning networks to analyze skin lesion photographs or webcam frames on-device instantly and privately. Receive educational risk assessments, interactive explainability heatmaps (Grad-CAM), and professional AI-generated summaries powered by Google Gemini 2.5.
        </p>
        """, unsafe_allow_html=True)

        st.write("") # small spacing

        if st.button("Start Image Analysis", key="btn_hero_upload"):
            st.session_state.navigation = "Analyze Image"
            st.rerun()

        if st.button("Live Webcam Screening", key="btn_hero_camera"):
            st.session_state.navigation = "Live Webcam"
            st.rerun()

        st.markdown("""
        <a class="btn-github-cta" href="https://github.com/AdityaSingh/VisionScan" target="_blank">
            <span class="material-icons" style="margin-right: 8px; font-size: 20px; vertical-align: middle;">code</span>
            View on GitHub
        </a>
        """, unsafe_allow_html=True)

    with col_hero_right:
        st.image("assets/hero.png", use_container_width=True)

    # ── 2. Trust Metrics Section ─────────────────────────────────────
    st.markdown("""
    <div class="section-title-wrapper">
        <h3 class="section-headline">Proven Technical Sophistication</h3>
    </div>
    <div class="metrics-grid">
        <div class="metric-card">
            <span class="material-icons metric-icon">biotech</span>
            <div class="metric-num">85.6%</div>
            <div class="metric-title">Validation Accuracy</div>
            <div class="metric-desc">High scientific credibility backed by rigorous calibration testing.</div>
        </div>
        <div class="metric-card">
            <span class="material-icons metric-icon">photo_library</span>
            <div class="metric-num">2,090+</div>
            <div class="metric-title">Training Images</div>
            <div class="metric-desc">Robust dataset optimization across diverse lesion types.</div>
        </div>
        <div class="metric-card">
            <span class="material-icons metric-icon">videocam</span>
            <div class="metric-num">Real-Time</div>
            <div class="metric-title">Webcam Detection</div>
            <div class="metric-desc">Immediate interactive scanner speed with on-device frames inference.</div>
        </div>
        <div class="metric-card">
            <span class="material-icons metric-icon">volunteer_activism</span>
            <div class="metric-num">100% Free</div>
            <div class="metric-title">Open Source</div>
            <div class="metric-desc">Transparent code, open-access development, and no user tracking.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 3. Features Section ──────────────────────────────────────────
    st.markdown("""
    <div class="section-title-wrapper">
        <h3 class="section-headline">Core Engine Features</h3>
    </div>
    <div class="features-grid">
        <div class="feature-card">
            <span class="material-icons feature-icon">upload_file</span>
            <div class="feature-title">Upload Image Analysis</div>
            <div class="feature-desc">Automated skin health evaluation with immediate risk classification and statistical indicators.</div>
        </div>
        <div class="feature-card">
            <span class="material-icons feature-icon">camera_alt</span>
            <div class="feature-title">Live Webcam Screening</div>
            <div class="feature-desc">Interactive real-time capture from your device's camera for immediate localized skin assessment.</div>
        </div>
        <div class="feature-card">
            <span class="material-icons feature-icon">psychology</span>
            <div class="feature-title">Explainable AI (Grad-CAM)</div>
            <div class="feature-desc">Visual activation heatmaps identifying the exact clinical regions of interest analyzed by the neural network.</div>
        </div>
        <div class="feature-card">
            <span class="material-icons feature-icon">smart_toy</span>
            <div class="feature-title">Gemini-Powered Explanations</div>
            <div class="feature-desc">Surgical narrative medical context, risk breakdowns, and dynamic interactive chatbot consultations.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 4. How It Works Section ─────────────────────────────────────
    st.markdown("""
    <div class="section-title-wrapper">
        <h3 class="section-headline">Surgical Scanner Workflow</h3>
    </div>
    <div class="steps-flow">
        <div class="step-card">
            <div class="step-num">1</div>
            <div class="step-title">Upload or Scan</div>
            <div class="step-desc">Provide a high-contrast dermoscopic photograph or capture a live webcam frame.</div>
        </div>
        <div class="step-card">
            <div class="step-num">2</div>
            <div class="step-title">Local AI Analysis</div>
            <div class="step-desc">The neural network instantly processes the image locally on-device, preserving privacy.</div>
        </div>
        <div class="step-card">
            <div class="step-num">3</div>
            <div class="step-title">Download Report</div>
            <div class="step-desc">Receive explainability maps, a clinical disclaimer, and download a PDF report.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 5. Open-Source Mission Section ─────────────────────────────
    st.markdown("""
    <div class="mission-banner">
        <span class="material-icons mission-logo">public</span>
        <div class="mission-content">
            <h4 class="mission-title">Built for Global Impact</h4>
            <p class="mission-desc">
                VisionScan is a free, community-driven educational platform designed to make advanced dermatological AI screening open and accessible worldwide. By hosting our project under the MIT License, we encourage academic research, model replication, and collaborative scientific transparency to help people better understand potentially concerning skin lesions.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 6. Medical Disclaimer Card ──────────────────────────────────
    st.markdown("""
    <div class="disclaimer-alert-box">
        <span class="material-icons disclaimer-alert-icon">info</span>
        <div class="disclaimer-alert-content">
            <strong>Medical Disclaimer:</strong> This application is a specialized deep learning educational tool and is not a substitute for professional medical diagnosis, clinical evaluation, or treatment. Any concerning skin lesion should be evaluated immediately by a qualified dermatologist.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 7. Model Performance Section ────────────────────────────────
    st.markdown("""
    <div class="section-title-wrapper">
        <h3 class="section-headline">🔬 Model Evaluation & Diagnostic Curves</h3>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Expand Model Evaluation Metrics & Curves", expanded=False):
        mf = RESULTS / "metrics.json"
        if mf.exists():
            m = json.loads(mf.read_text())
            top = st.columns(4)
            for col, (k, l) in zip(top, [("val_accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1_score", "F1")]):
                col.metric(l, f"{m.get(k, 0):.2%}")

            extra = st.columns(4)
            for col, (k, l) in zip(extra, [("specificity", "Specificity"), ("roc_auc", "ROC AUC"), ("pr_auc", "PR AUC"), ("val_loss", "Val Loss")]):
                v = m.get(k)
                if v is not None:
                    col.metric(l, f"{v:.4f}" if k == "val_loss" else f"{v:.2%}")

        st.markdown("---")
        tabs = st.tabs(["Training Curves", "Confusion Matrix", "ROC", "PR + Calibration"])

        plots = {
            0: "training_curves.png",
            1: "confusion_matrix.png",
            2: "roc_curve.png",
        }
        for i, name in plots.items():
            with tabs[i]:
                p = RESULTS / name
                if p.exists():
                    st.image(str(p), use_container_width=True)
                else:
                    st.info(f"{name} plot not available yet.")

        with tabs[3]:
            c1, c2 = st.columns(2)
            for col, name in [(c1, "pr_curve.png"), (c2, "reliability_diagram.png")]:
                p = RESULTS / name
                if p.exists():
                    col.image(str(p), use_container_width=True)


def page_upload(model: tf.keras.Model) -> None:
    st.title("Analyze Image")
    st.write("Upload a dermoscopic image or clear close-up of a skin lesion for automated evaluation.")

    col_upl, col_res = st.columns([5, 5])

    with col_upl:
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded is not None:
            # 🛡️ Secure Upload Validation Pass
            is_valid, err_msg, loaded_image = validate_image_upload(uploaded, max_mb=MAX_UPLOAD_MB)
            if not is_valid:
                st.error(f"Upload Rejected: {err_msg}")
                return

            # 🛡️ Stripping EXIF Metadata & Sanitising filenames
            clean_image = strip_metadata(loaded_image)
            safe_name = secure_filename(uploaded.name)
            tmp = RESULTS / f"temp_{safe_name}"
            clean_image.save(tmp, format="JPEG", quality=95)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(clean_image, use_container_width=True, caption="Original Lesion Image")
            st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is None:
        return

    with col_res:
        with st.spinner("Processing local AI inference..."):
            rgb, batch = preprocess(clean_image)
            pred = run_async_inference(model, batch)

            conv_layer = find_last_conv_layer(model)
            cam_path = None
            if conv_layer:
                try:
                    heatmap = compute_heatmap(batch, model, conv_layer)
                    blended = overlay(rgb, heatmap)
                    cam_path = RESULTS / f"gradcam_{safe_name}"
                    Image.fromarray(blended).save(cam_path)
                except Exception as e:
                    log.warning("Grad-CAM overlay failed: %s", e)

        # Save to session state for assistant chatbot context
        st.session_state.last_prediction = {
            "label": pred.label,
            "confidence": pred.confidence,
            "risk": pred.risk.value,
            "certainty": pred.certainty.value,
            "recommendation": pred.recommendation,
        }

        # Colored Risk Banner
        colour = _hex(pred.risk)
        display = "Inconclusive" if pred.risk is RiskLevel.INCONCLUSIVE else pred.label
        st.markdown(
            f"<div class='banner' style='background:{colour}'>"
            f"<h2>{display} — {pred.confidence:.1%}</h2>"
            f"<p>{pred.risk.value} · {pred.certainty.value}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Confidence Bar Component
        st.markdown(f"**Classification Confidence:** {pred.confidence:.1%}")
        st.progress(float(pred.confidence))

        # Styled recommendation container
        st.markdown(f"<div class='recommendation-alert'><strong>Recommendation:</strong> {pred.recommendation}</div>", unsafe_allow_html=True)

        _log(safe_name, pred)

        # Dynamic PDF Report Downloader
        gemini_active = load_gemini_client() is not None
        if st.button("Generate PDF Report"):
            with st.spinner("Compiling Clinical Report..."):
                narrative = None
                if gemini_active:
                    narrative = generate_pdf_narrative(
                        prediction=pred.label,
                        confidence=pred.confidence,
                        risk_level=pred.risk.value,
                        certainty=pred.certainty.value,
                        recommendation=pred.recommendation,
                        session_id=st.context.headers.get("cookie", "default"),
                    )
                pdf = generate_pdf(tmp, pred.label, pred.confidence, pred.risk.value, pred.recommendation, cam_path, narrative=narrative)
            with open(pdf, "rb") as f:
                st.download_button("Download PDF Report", data=f, file_name=f"VisionScan_{safe_name}.pdf", mime="application/pdf")

    # Grad-CAM Attention Map & LLM Explanations Bottom Row
    st.markdown("---")
    st.markdown("### Explainable AI & Detailed Narratives")

    col_cam_view, col_gemini_view = st.columns([5, 5])
    with col_cam_view:
        if cam_path and cam_path.exists():
            st.markdown("#### AI Attention Map (Grad-CAM)")
            st.image(str(cam_path), use_container_width=True, caption="Gradient focus highlights pixel activation nodes")
        else:
            st.info("Grad-CAM attention maps are only available for convolutional structures.")

    with col_gemini_view:
        if gemini_active:
            st.markdown("#### Educational AI Narrative")
            lang = st.selectbox("Preferred Language", LANGUAGES, index=0)
            if st.checkbox("Generate Deep Explanation", value=True):
                with st.spinner("Synthesising details..."):
                    explanation = generate_explanation(
                        prediction=pred.label,
                        confidence=pred.confidence,
                        risk_level=pred.risk.value,
                        certainty=pred.certainty.value,
                        recommendation=pred.recommendation,
                        language=lang,
                        session_id=st.context.headers.get("cookie", "default"),
                    )
                st.markdown(explanation)
                st.session_state.last_explanation = explanation
            st.info("Gemini AI is currently offline. Review your sidebar .env key configuration.")

    # ── Recent Analyses (History) Section ────────────────────────────
    st.markdown("---")
    with st.expander("⏳ Recent Analyses (History)", expanded=False):
        df = get_prediction_history_df()
        if not df.empty:
            st.write(f"Total entries: **{len(df)}**")
            st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV History", data=csv_bytes, file_name="prediction_history.csv", mime="text/csv", key="btn_upload_csv_history")
        else:
            st.info("No predictions logged yet inside SQLite.")


def page_camera(model: tf.keras.Model) -> None:
    st.title("Live Webcam")
    st.write("Scan skin lesions in real-time using your device camera capture.")

    buf = st.camera_input("Capture Closeup Scan")
    if buf is None:
        return

    # Buffer verification
    is_valid, err_msg, loaded_image = validate_image_upload(buf, max_mb=MAX_UPLOAD_MB)
    if not is_valid:
        st.error(f"Camera capture rejected: {err_msg}")
        return

    # Stripping privacy data
    clean_image = strip_metadata(loaded_image)
    tmp = RESULTS / "temp_webcam.jpg"
    clean_image.save(tmp, format="JPEG", quality=90)

    st.markdown("---")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.subheader("Captured Lesion Image")
        st.image(clean_image, use_container_width=True)

    with col_c2:
        with st.spinner("Analysing frame..."):
            _, batch = preprocess(clean_image)
            pred = run_async_inference(model, batch)

        # Save to session state for chatbot context
        st.session_state.last_prediction = {
            "role": "user",
            "content": f"Classification Context: {pred.label} ({pred.confidence:.1%})",
        }

        colour = _hex(pred.risk)
        display = "Inconclusive" if pred.risk is RiskLevel.INCONCLUSIVE else pred.label
        st.markdown(
            f"<div class='banner' style='background:{colour}'>"
            f"<h2>{display} — {pred.confidence:.1%}</h2>"
            f"<p>{pred.risk.value}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(f"**Inference Confidence Interval:** {pred.confidence:.1%}")
        st.progress(float(pred.confidence))

        st.markdown(f"<div class='recommendation-alert'><strong>Recommendation:</strong> {pred.recommendation}</div>", unsafe_allow_html=True)

        # Live Gemini Assistant explanation fallback
        gemini_active = load_gemini_client() is not None
        if gemini_active:
            with st.expander("AI Educational Assessment Details"):
                with st.spinner("Generating..."):
                    explanation = generate_explanation(
                        prediction=pred.label,
                        confidence=pred.confidence,
                        risk_level=pred.risk.value,
                        certainty=pred.certainty.value,
                        recommendation=pred.recommendation,
                        session_id=st.context.headers.get("cookie", "default"),
                    )
                st.markdown(explanation)

    _log("webcam", pred)


def page_batch(model: tf.keras.Model) -> None:
    st.title("Batch Testing")
    st.write("Process and analyze entire folders of dermoscopic images sequentially.")

    folder_raw = st.text_input("Folder Path:", value="test_images")
    folder = sanitize_text(folder_raw)

    if not st.button("Run Batch Assessment"):
        return

    p = Path(folder).resolve()
    # Path Sandboxing: ensure folder is strictly located inside current directory
    try:
        p.relative_to(Path.cwd().resolve())
    except ValueError:
        st.error("Access Denied: Directories outside the project workspace are blocked for safety.")
        log.warning("Blocked traversal attempt to path: %s", p)
        return

    if not p.is_dir():
        st.error("Directory not found or invalid.")
        return

    # Count matching files to protect limits
    total_files = len([f for f in p.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if total_files > MAX_BATCH_IMAGES:
        st.error(f"Batch limit exceeded: Folder contains {total_files} images. The maximum limit is {MAX_BATCH_IMAGES} images.")
        return

    with st.spinner(f"Processing folder '{folder}'..."):
        future = _INFERENCE_POOL.submit(run_batch, model, p)
        df = future.result()

    if df is None or df.empty:
        st.warning("No valid images detected inside the folder.")
        return

    st.success(f"Batch analysis complete. Processed {len(df)} images.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Benign Scans", int((df["Prediction"] == "Benign").sum()))
    col2.metric("Malignant Scans", int((df["Prediction"] == "Malignant").sum()))
    col3.metric("Average Confidence", f"{df['Confidence'].mean():.1%}")

    st.dataframe(df, use_container_width=True)

    batch_dir = Path("results/batch_testing")
    st.subheader("Distribution Analytics")
    ch1, ch2 = st.columns(2)
    for col, name in [(ch1, "prediction_distribution.png"), (ch2, "risk_distribution.png")]:
        img = batch_dir / name
        if img.exists():
            col.image(str(img), use_container_width=True)

    csv = batch_dir / "batch_results.csv"
    if csv.exists():
        st.download_button("Download CSV Results", data=csv.read_bytes(), file_name="batch_results.csv", mime="text/csv")


def page_assistant() -> None:
    st.title("AI Assistant")
    st.write("Converse with our educational assistant about skin lesions, risk factors, or model metrics.")

    context = st.session_state.get("last_prediction")
    if context:
        st.markdown(
            f"<div style='background-color: #F0F2F5; padding: 12px 18px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #DADDE1;'>"
            f"<strong>Active Diagnostic Context Loaded:</strong> "
            f"{context['label']} ({context['confidence']:.1%}) · {context['risk']}"
            f"</div>",
            unsafe_allow_html=True
        )

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display past history using our sleek dual bubbles styling
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        role_class = "user" if msg["role"] == "user" else "assistant"
        st.markdown(f"<div class='chat-bubble {role_class}'>{msg['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Suggested Questions Row
    st.markdown("Suggested Questions:")
    s1, s2, s3 = st.columns(3)
    suggested = ""
    if s1.button("What does risk stratification mean?", key="s1"):
        suggested = "What does risk stratification mean?"
    if s2.button("What is a temperature-calibrated probability?", key="s2"):
        suggested = "What is a temperature-calibrated probability?"
    if s3.button("When should I consult a dermatologist?", key="s3"):
        suggested = "When should I consult a dermatologist?"

    # User chat input
    user_q_raw = st.chat_input("Ask a question...")
    query = sanitize_text(user_q_raw) if user_q_raw else suggested

    if query:
        # Display immediately on user send
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.spinner("Consulting dermatology database..."):
            ans = answer_user_question(
                question=query,
                current_prediction=context,
                chat_history=st.session_state.chat_history[:-1],
                session_id=st.context.headers.get("cookie", "default"),
            )
        st.session_state.chat_history.append({"role": "assistant", "content": ans})
        st.rerun()


def page_metrics() -> None:
    st.title("Model Performance")

    mf = RESULTS / "metrics.json"
    if mf.exists():
        m = json.loads(mf.read_text())
        top = st.columns(4)
        for col, (k, l) in zip(top, [("val_accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1_score", "F1")]):
            col.metric(l, f"{m.get(k, 0):.2%}")

        extra = st.columns(4)
        for col, (k, l) in zip(extra, [("specificity", "Specificity"), ("roc_auc", "ROC AUC"), ("pr_auc", "PR AUC"), ("val_loss", "Val Loss")]):
            v = m.get(k)
            if v is not None:
                col.metric(l, f"{v:.4f}" if k == "val_loss" else f"{v:.2%}")

    st.markdown("---")
    tabs = st.tabs(["Training Curves", "Confusion Matrix", "ROC", "PR + Calibration"])

    plots = {
        0: "training_curves.png",
        1: "confusion_matrix.png",
        2: "roc_curve.png",
    }
    for i, name in plots.items():
        with tabs[i]:
            p = RESULTS / name
            if p.exists():
                st.image(str(p), use_container_width=True)
            else:
                st.info(f"{name} plot not available yet.")

    with tabs[3]:
        c1, c2 = st.columns(2)
        for col, name in [(c1, "pr_curve.png"), (c2, "reliability_diagram.png")]:
            p = RESULTS / name
            if p.exists():
                col.image(str(p), use_container_width=True)


def page_history() -> None:
    st.title("Prediction History")
    df = get_prediction_history_df()
    if not df.empty:
        st.write(f"Total entries: **{len(df)}**")
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV History", data=csv_bytes, file_name="prediction_history.csv", mime="text/csv")
    else:
        st.info("No predictions logged yet inside SQLite.")


def page_about() -> None:
    st.title("About VisionScan")
    st.markdown("""
### Mission

Make AI-powered skin lesion risk assessment accessible, transparent,
and reproducible for everyone.

### Technology Stack

| Component | Stack |
|---|---|
| Deep Learning | TensorFlow, Keras |
| Architectures | MobileNetV2 / EfficientNetV2 / ConvNeXt |
| Explainability | Grad-CAM |
| LLM Reasoning | Google Gemini 2.5 Flash |
| Database Engine | SQLite (WAL Enabled) |
| Async Worker | ThreadPoolExecutor |
| Frontend | Streamlit |
| Reports | ReportLab |
| CI/CD | GitHub Actions, Docker |
    """)

    st.markdown("---")
    with st.expander("🔬 Developer Tools & Batch Testing", expanded=False):
        st.write("Process and analyze entire folders of dermoscopic images sequentially.")
        if st.checkbox("Enable Developer Batch Testing Mode", value=False, key="chk_dev_batch_testing"):
            # Load the model dynamically on-demand!
            with st.spinner("Initializing neural engine..."):
                model = _load()
            if model is None:
                st.error("Model not found. Run python train.py first.")
            else:
                page_batch(model)


# ── Main Navigation Runner ───────────────────────────────────────────

def main() -> None:
    st.sidebar.markdown("## VisionScan")
    st.sidebar.caption("Intelligent skin health screening")
    st.sidebar.markdown("---")

    pages = {
        "Home": page_home,
        "Analyze Image": page_upload,
        "Live Webcam": page_camera,
        "AI Assistant": page_assistant,
        "About": page_about,
    }

    # Custom session persistence for redirection triggers
    if "navigation" not in st.session_state:
        st.session_state.navigation = "Home"

    # Match sidebar menu selection
    sidebar_index = list(pages.keys()).index(st.session_state.navigation) if st.session_state.navigation in pages else 0
    choice = st.sidebar.radio("Navigate", list(pages.keys()), index=sidebar_index, label_visibility="collapsed")
    st.session_state.navigation = choice

    # Elegant Compact Sidebar Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="padding: 10px 0; text-align: left; font-family: 'Inter', sans-serif;">
        <p style="margin: 0; font-size: 0.8rem; color: #65676B; font-weight: 500;">v2.0.0</p>
        <p style="margin: 4px 0; font-size: 0.75rem; color: #65676B;">Open Source • MIT License</p>
        <a href="https://github.com/AdityaSingh/VisionScan" target="_blank" style="font-size: 0.75rem; color: #1877F2; text-decoration: none; font-weight: 600;">
            <span class="material-icons" style="font-size: 14px; vertical-align: middle; margin-right: 2px;">code</span>
            GitHub Repository
        </a>
    </div>
    """, unsafe_allow_html=True)

    # Initialise secure WAL sqlite database schema on launch
    init_db()

    page_fn = pages[choice]
    # Pages that do not need the deep learning model (enables instant landing page loads)
    if choice in ("Home", "AI Assistant", "About"):
        page_fn()
    else:
        # Load deep learning model dynamically only when entering prediction pipelines
        model = _load()
        if model is None:
            st.error(f"Model not found at `{MODEL_PATH}`. Run `python train.py` first.")
            st.stop()
        page_fn(model)

    st.markdown(f"<div class='footer'>{DISCLAIMER}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()


# ── Vercel Build Compatibility ────────────────────────────────────────
# Dummy WSGI app entrypoint to satisfy Vercel serverless function checks
def app(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html; charset=utf-8')])
    return [b"<h1>VisionScan Global</h1><p>Streamlit is active. Please launch locally using 'streamlit run app.py' or view the deployed instance on Streamlit Community Cloud.</p>"]
