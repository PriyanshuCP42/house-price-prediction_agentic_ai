"""
Enhanced PDF Report Generator — Extends Milestone 1 PDF with
agent advisory sections: market context, risk assessment, and recommendation.
"""

import re
import datetime
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph,
    PageBreak, KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER


def _clean_text(text: str) -> str:
    """Strip markdown formatting and HTML tags for safe PDF rendering."""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Convert markdown bold to ReportLab bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # Convert markdown italic
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    # Remove markdown headers (##)
    text = re.sub(r'^#{1,4}\s*', '', text, flags=re.MULTILINE)
    # Escape ampersands that are not already entities
    text = text.replace('&', '&amp;')
    # Restore any double-escaped entities
    text = text.replace('&amp;amp;', '&amp;')
    return text.strip()


def _safe_paragraph(text: str, style) -> Paragraph:
    """Create a Paragraph that won't crash on bad markup."""
    try:
        return Paragraph(_clean_text(text), style)
    except Exception:
        # Fallback: strip all tags
        plain = re.sub(r'<[^>]+>', '', text or "")
        return Paragraph(plain, style)


def generate_advisory_pdf(report) -> BytesIO:
    """Generate a professional PDF advisory report from the validated report schema."""

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
    )

    styles = getSampleStyleSheet()
    page_width = letter[0] - 1.5 * inch  # usable width

    # ----- Custom Styles -----
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"],
        fontSize=18, textColor=colors.HexColor("#1e293b"),
        spaceAfter=4, alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#64748b"),
        alignment=TA_CENTER, spaceAfter=14,
    )
    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Heading2"],
        fontSize=13, textColor=colors.HexColor("#1e3a5f"),
        spaceBefore=18, spaceAfter=8,
        borderWidth=0, borderPadding=0,
    )
    body_style = ParagraphStyle(
        "BodyStyle", parent=styles["Normal"],
        fontSize=9.5, leading=14, textColor=colors.HexColor("#334155"),
        spaceAfter=2,
    )
    bullet_style = ParagraphStyle(
        "BulletStyle", parent=body_style,
        leftIndent=16, bulletIndent=6,
        spaceBefore=1, spaceAfter=1,
    )
    small_style = ParagraphStyle(
        "SmallStyle", parent=styles["Normal"],
        fontSize=7.5, textColor=colors.HexColor("#94a3b8"),
    )
    disclaimer_heading = ParagraphStyle(
        "DiscHead", parent=heading_style,
        fontSize=10, textColor=colors.HexColor("#991b1b"),
    )
    disclaimer_body = ParagraphStyle(
        "DiscBody", parent=body_style,
        fontSize=7, textColor=colors.HexColor("#64748b"),
        leading=10,
    )

    elements = []

    # ===== HEADER =====
    elements.append(Paragraph("AI Property Decision Copilot", title_style))
    elements.append(Paragraph("Professional Advisory Report", subtitle_style))
    elements.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}  |  "
        f"{report.consultation.client_mode.title()} Mode  |  {report.consultation.objective}  |  King County, WA",
        small_style,
    ))
    elements.append(Spacer(1, 14))

    # ===== EXECUTIVE SUMMARY TABLE =====
    val = report.valuation
    rec = report.recommendation.replace("_", " ")

    col_w = page_width / 4
    summary_data = [
        [
            Paragraph("<b>Predicted Price</b>", ParagraphStyle("th", parent=body_style, fontSize=9, textColor=colors.white, alignment=TA_CENTER)),
            Paragraph("<b>Confidence</b>", ParagraphStyle("th", parent=body_style, fontSize=9, textColor=colors.white, alignment=TA_CENTER)),
            Paragraph("<b>Neighborhood</b>", ParagraphStyle("th", parent=body_style, fontSize=9, textColor=colors.white, alignment=TA_CENTER)),
            Paragraph("<b>Recommendation</b>", ParagraphStyle("th", parent=body_style, fontSize=9, textColor=colors.white, alignment=TA_CENTER)),
        ],
        [
            Paragraph(f"<b>${val.predicted_price:,.0f}</b>", ParagraphStyle("tv", parent=body_style, fontSize=14, alignment=TA_CENTER)),
            Paragraph(f"<b>{val.confidence:.1f}%</b>", ParagraphStyle("tv", parent=body_style, fontSize=14, alignment=TA_CENTER)),
            Paragraph(f"<b>{report.neighborhood.overall_score}/100</b>", ParagraphStyle("tv", parent=body_style, fontSize=14, alignment=TA_CENTER)),
            Paragraph(f"<b>{rec}</b>", ParagraphStyle("tv", parent=body_style, fontSize=14, alignment=TA_CENTER)),
        ],
        [
            Paragraph(f"Range: ${val.price_low:,.0f} - ${val.price_high:,.0f}", ParagraphStyle("ts", parent=small_style, alignment=TA_CENTER)),
            Paragraph("High" if val.confidence >= 70 else "Medium" if val.confidence >= 50 else "Low", ParagraphStyle("ts", parent=small_style, alignment=TA_CENTER)),
            Paragraph(report.neighborhood.market_heat, ParagraphStyle("ts", parent=small_style, alignment=TA_CENTER)),
            Paragraph(f"Risk: {report.risk_level}", ParagraphStyle("ts", parent=small_style, alignment=TA_CENTER)),
        ],
    ]

    summary_table = Table(summary_data, colWidths=[col_w] * 4)
    rec_color = (
        colors.HexColor("#d1fae5") if rec in ("STRONG BUY", "BUY")
        else colors.HexColor("#fef3c7") if rec == "HOLD"
        else colors.HexColor("#fee2e2")
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (3, 1), (3, 2), rec_color),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 16))

    # ===== ADVISORY REPORT SECTIONS =====
    report_text = report.advisory_markdown or ""
    sections = report_text.split("## ")

    for section in sections:
        if not section.strip():
            continue
        lines = section.strip().split("\n")
        section_title = lines[0].strip()
        section_body_lines = lines[1:]

        if section_title:
            elements.append(Paragraph(_clean_text(section_title), heading_style))

        for line in section_body_lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("- "):
                elements.append(_safe_paragraph("&bull; " + line[2:], bullet_style))
            elif line.startswith("* "):
                elements.append(_safe_paragraph("&bull; " + line[2:], bullet_style))
            else:
                elements.append(_safe_paragraph(line, body_style))

    # ===== NEGOTIATION STRATEGY =====
    neg = report.negotiation
    elements.append(Paragraph("Negotiation Strategy", heading_style))

    neg_data = [
        [
            Paragraph("<b>Anchor Price</b>", ParagraphStyle("nh", parent=body_style, fontSize=9, textColor=colors.white)),
            Paragraph("<b>Target Price</b>", ParagraphStyle("nh", parent=body_style, fontSize=9, textColor=colors.white)),
            Paragraph("<b>Walk-Away Price</b>", ParagraphStyle("nh", parent=body_style, fontSize=9, textColor=colors.white)),
        ],
        [
            Paragraph(f"${neg.anchor_price:,.0f}", ParagraphStyle("nv", parent=body_style, fontSize=12, alignment=TA_CENTER)),
            Paragraph(f"${neg.target_price:,.0f}", ParagraphStyle("nv", parent=body_style, fontSize=12, alignment=TA_CENTER)),
            Paragraph(f"${neg.walk_away_price:,.0f}", ParagraphStyle("nv", parent=body_style, fontSize=12, alignment=TA_CENTER)),
        ],
    ]
    neg_table = Table(neg_data, colWidths=[page_width / 3] * 3)
    neg_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(neg_table)
    elements.append(Spacer(1, 6))

    if neg.strategy_summary:
        elements.append(_safe_paragraph(neg.strategy_summary, body_style))
    if neg.leverage_points:
        elements.append(Paragraph("<b>Leverage Points</b>", body_style))
        for lp in neg.leverage_points:
            elements.append(_safe_paragraph("&bull; " + lp, bullet_style))
    if neg.caution_points:
        elements.append(Paragraph("<b>Caution Points</b>", body_style))
        for cp in neg.caution_points:
            elements.append(_safe_paragraph("&bull; " + cp, bullet_style))

    # ===== RISK FACTOR TABLE =====
    if report.risk_factors:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Risk Factor Details", heading_style))

        risk_header = [
            Paragraph("<b>Factor</b>", ParagraphStyle("rh", parent=body_style, fontSize=8.5, textColor=colors.white)),
            Paragraph("<b>Severity</b>", ParagraphStyle("rh", parent=body_style, fontSize=8.5, textColor=colors.white)),
            Paragraph("<b>Score</b>", ParagraphStyle("rh", parent=body_style, fontSize=8.5, textColor=colors.white)),
            Paragraph("<b>Mitigation</b>", ParagraphStyle("rh", parent=body_style, fontSize=8.5, textColor=colors.white)),
        ]
        risk_data = [risk_header]
        for rf in report.risk_factors:
            risk_data.append([
                Paragraph(rf.factor, ParagraphStyle("rv", parent=body_style, fontSize=8)),
                Paragraph(rf.severity, ParagraphStyle("rv", parent=body_style, fontSize=8)),
                Paragraph(f"{rf.score:.1f}/12.5", ParagraphStyle("rv", parent=body_style, fontSize=8)),
                Paragraph(rf.mitigation, ParagraphStyle("rv", parent=body_style, fontSize=8)),
            ])

        risk_table = Table(risk_data, colWidths=[1.4 * inch, 0.8 * inch, 0.7 * inch, page_width - 2.9 * inch])
        risk_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 0), (2, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        elements.append(risk_table)

    # ===== NEIGHBORHOOD SCORECARD =====
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Neighborhood Scorecard", heading_style))

    hood = report.neighborhood
    hood_data = [
        [
            Paragraph("<b>Metric</b>", ParagraphStyle("hh", parent=body_style, fontSize=9, textColor=colors.white)),
            Paragraph("<b>Score</b>", ParagraphStyle("hh", parent=body_style, fontSize=9, textColor=colors.white)),
        ],
        [Paragraph("Overall", body_style), Paragraph(f"{hood.overall_score}/100", body_style)],
        [Paragraph("Livability", body_style), Paragraph(f"{hood.livability_score}/100", body_style)],
        [Paragraph("Liquidity", body_style), Paragraph(f"{hood.liquidity_score}/100", body_style)],
        [Paragraph("Upside Potential", body_style), Paragraph(f"{hood.upside_score}/100", body_style)],
        [Paragraph("Rental Demand", body_style), Paragraph(f"{hood.rental_demand_score}/100", body_style)],
        [Paragraph("Pricing Power", body_style), Paragraph(f"{hood.pricing_power_score}/100", body_style)],
        [Paragraph("Market Heat", body_style), Paragraph(hood.market_heat.title(), body_style)],
    ]

    hood_table = Table(hood_data, colWidths=[2.5 * inch, 1.5 * inch])
    hood_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(hood_table)

    if hood.narrative:
        elements.append(Spacer(1, 4))
        elements.append(_safe_paragraph(hood.narrative, body_style))

    # ===== KNOWLEDGE SOURCES =====
    if report.rag_sources:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Knowledge Base Sources", heading_style))
        for src in report.rag_sources:
            elements.append(_safe_paragraph("&bull; " + src, bullet_style))

    # ===== DISCLAIMER =====
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Disclaimer", disclaimer_heading))
    elements.append(_safe_paragraph(
        report.disclaimers or (
            "This report is AI-generated for educational purposes only. "
            "It does not constitute financial, legal, or professional real estate advice. "
            "Always consult qualified professionals before making investment decisions."
        ),
        disclaimer_body,
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer
