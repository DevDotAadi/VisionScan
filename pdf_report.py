"""PDF assessment report generator for VisionScan Global.

Produces a downloadable clinical-style educational report containing
the uploaded image, prediction, Grad-CAM overlay, Gemini narrative, and disclaimer.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

log = logging.getLogger(__name__)

DISCLAIMER = (
    "DISCLAIMER — This application is an educational AI tool and is not a "
    "substitute for professional medical diagnosis. Any concerning skin "
    "lesion should be evaluated by a qualified dermatologist."
)


def generate(
    image_path: str | Path,
    prediction: str,
    confidence: float,
    risk: str,
    recommendation: str,
    gradcam_path: str | Path | None = None,
    output_path: str | Path = "results/report.pdf",
    narrative: str | None = None,
) -> Path:
    """Build a PDF report and return the output path."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output), pagesize=letter,
        leftMargin=72, rightMargin=72, topMargin=72, bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Heading1"],
        alignment=TA_CENTER, spaceAfter=20,
    )
    result_colour = colors.red if prediction == "Malignant" else colors.green
    result_style = ParagraphStyle(
        "ResultText", parent=styles["Normal"],
        textColor=result_colour, fontSize=12, spaceAfter=10,
    )
    note_style = ParagraphStyle(
        "Note", parent=styles["Italic"],
        fontSize=9, textColor=colors.gray, spaceBefore=24,
    )
    narrative_style = ParagraphStyle(
        "Narrative", parent=styles["Normal"],
        fontSize=10, leading=14, spaceBefore=8, spaceAfter=12,
    )

    elements: list = []

    elements.append(Paragraph("VisionScan AI — Skin Lesion Assessment Report", title_style))
    elements.append(Paragraph(
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Assessment Results", styles["Heading2"]))
    elements.append(Paragraph(f"<b>Prediction:</b> {prediction}", result_style))
    elements.append(Paragraph(f"<b>Confidence:</b> {confidence:.2%}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Risk Level:</b> {risk}", styles["Normal"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"<b>Recommendation:</b> {recommendation}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    if narrative:
        elements.append(Paragraph("<b>Gemini AI Report Narrative:</b>", styles["Normal"]))
        elements.append(Paragraph(narrative, narrative_style))
        elements.append(Spacer(1, 10))

    img_path = Path(image_path)
    if img_path.exists():
        try:
            img = PILImage.open(img_path)
            aspect = img.height / img.width
            w = 250
            elements.append(Paragraph("Submitted Image:", styles["Heading3"]))
            elements.append(RLImage(str(img_path), width=w, height=w * aspect))
            elements.append(Spacer(1, 16))
        except Exception:
            log.warning("Could not embed original image in PDF")

    if gradcam_path:
        gc = Path(gradcam_path)
        if gc.exists():
            try:
                img = PILImage.open(gc)
                aspect = img.height / img.width
                w = 250
                elements.append(Paragraph("AI Attention Map (Grad-CAM):", styles["Heading3"]))
                elements.append(RLImage(str(gc), width=w, height=w * aspect))
                elements.append(Spacer(1, 16))
            except Exception:
                log.warning("Could not embed Grad-CAM image in PDF")

    elements.append(Spacer(1, 30))
    elements.append(Paragraph(DISCLAIMER, note_style))

    doc.build(elements)
    log.info("PDF report saved to %s", output)
    return output
