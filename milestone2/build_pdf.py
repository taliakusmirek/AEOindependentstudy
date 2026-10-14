import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))  # run from anywhere
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 Image, PageBreak, ListFlowable, ListItem)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from build_profound_csv import rows as prompt_rows

styles = getSampleStyleSheet()
title_style = ParagraphStyle('TitleCustom', parent=styles['Title'], textColor=colors.black, fontSize=20, spaceAfter=6)
h1 = ParagraphStyle('H1Custom', parent=styles['Heading1'], textColor=colors.black, fontSize=14, spaceBefore=18, spaceAfter=8)
h2 = ParagraphStyle('H2Custom', parent=styles['Heading2'], textColor=colors.black, fontSize=11.5, spaceBefore=10, spaceAfter=4)
body = ParagraphStyle('BodyCustom', parent=styles['BodyText'], fontSize=10, leading=14, textColor=colors.black)
small = ParagraphStyle('SmallCustom', parent=styles['BodyText'], fontSize=9, leading=12, textColor=colors.black)
subtitle = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=10.5, textColor=colors.black, spaceAfter=4)

doc = SimpleDocTemplate("design_doc.pdf", pagesize=letter,
                         topMargin=0.75*inch, bottomMargin=0.75*inch,
                         leftMargin=0.8*inch, rightMargin=0.8*inch)

story = []

# --- Title ---
story.append(Paragraph("AEO Citation Pipeline — Design Document", title_style))
story.append(Paragraph("Milestone 2: Problem Formalization and System Design", subtitle))
story.append(Paragraph("Natalia Kusmirek &nbsp;|&nbsp; Readings in Computer Science &nbsp;|&nbsp; Instructor: Prof. Maíra Marques Samary", subtitle))
story.append(Spacer(1, 14))

# --- Section 1: Taxonomy ---
story.append(Paragraph("1. Cohort Taxonomy", h1))
story.append(Paragraph(
    "The classifier developed in Milestone 3 sorts every domain that cites Boston College across AI answer "
    "engines into one of six mutually exclusive categories. The table below defines each category and the "
    "rule used to assign it.", body))
story.append(Spacer(1, 8))

taxonomy_data = [
    ["Category", "Rule"],
    ["BC-owned", "The domain is a Boston College-controlled property (bc.edu and subdomains, official BC social accounts, BC-run publications)."],
    ["Earned media", "The domain is a third-party news, magazine, or broadcast outlet that published independently about BC without BC controlling the content."],
    ["Education directory /\nranking site", "The domain's primary function is aggregating or ranking schools/programs (e.g., Niche, Peterson's, College Navigator, US News rankings pages)."],
    ["Forum / community", "The domain hosts user-generated discussion (Reddit, College Confidential, Quora, and similar indexed community content)."],
    ["Competitor institution", "The domain belongs to another college or university cited alongside or instead of BC (e.g., bu.edu, nd.edu, holycross.edu)."],
    ["Other", "The domain does not fit the above five categories (e.g., .gov sites, Wikipedia, employer/career sites)."],
]
taxonomy_table = Table(
    [[Paragraph(f"<b>{r[0]}</b>", body) if i == 0 else Paragraph(r[0], body), Paragraph(r[1], body)] for i, r in enumerate(taxonomy_data)],
    colWidths=[1.6*inch, 4.6*inch]
)
taxonomy_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.white),
    ('TEXTCOLOR', (0,0), (-1,0), colors.black),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('GRID', (0,0), (-1,-1), 0.75, colors.black),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('TOPPADDING', (0,0), (-1,-1), 6),
    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ('LEFTPADDING', (0,0), (-1,-1), 8),
]))
story.append(taxonomy_table)
story.append(Spacer(1, 6))
story.append(Paragraph(
    "<b>Edge case note:</b> US News is split by page type — its editorial/news articles about BC are labeled "
    "earned media, while its rankings pages are labeled education directory/ranking site. This split is applied "
    "consistently during manual labeling in Section 3.", small))

# --- Section 2: Architecture diagram ---
story.append(Paragraph("2. Pipeline Architecture", h1))
story.append(Paragraph(
    "The full system is a 5-stage pipeline. Stages 1-2 (Profound export, citation extractor) are built and "
    "populated in this milestone; the classifier (Stage 3) is built in Milestone 3; the tagging layer and "
    "report generator (Stages 4-5) are built in Milestone 4.", body))
story.append(Spacer(1, 8))
story.append(Image("architecture_diagram.png", width=6.4*inch, height=6.4*inch*819/2159))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<b>Stage 1 - Profound Export:</b> the 26 prompts in Section 3 are run through a personal Profound seat; "
    "citations are exported as a raw CSV with columns domain, source_url, prompt, engine.<br/>"
    "<b>Stage 2 - Citation Extractor:</b> parses the raw export, deduplicates rows, and normalizes domain and "
    "engine formatting.<br/>"
    "<b>Stage 3 - Classifier:</b> assigns each row one of the six taxonomy categories (rule-based baseline and "
    "a learned model, built in Milestone 3).<br/>"
    "<b>Stage 4 - Tagging Layer:</b> appends intent, cohort, and platform columns so weekly snapshots remain "
    "statistically comparable over time.<br/>"
    "<b>Stage 5 - Report Generator:</b> computes citation-share deltas between snapshots, flags material moves, "
    "and writes a formatted weekly brief.", small))

# --- Section 3: Prompt list ---
story.append(Paragraph("3. Full Prompt List (26 prompts)", h1))
story.append(Paragraph(
    "Prompts are split across three intent types to ensure the resulting citation set reflects how real users "
    "query AI answer engines about Boston College. Informational and comparison prompts name Boston College "
    "directly, since these intents presuppose the user already knows to ask about BC specifically. Recommendation "
    "prompts deliberately do not name BC; they are category-level queries (Jesuit/Catholic education, rigorous "
    "academics, holistic education, undergraduate education) sourced from Profound's auto-generated prompt set "
    "for BC's brand profile. This tests whether BC is cited organically when a user has not already named it, "
    "which is the more realistic test of recommendation-intent citation behavior and the stronger methodological "
    "choice for this category. Each prompt is also tagged with a Topic and one or two target Personas "
    "(Prospective Student, and Parent of Prospective Student where the topic is one a parent would plausibly "
    "search), matching the metadata loaded into the Profound prompt set.", body))
story.append(Spacer(1, 8))

# small helper to shorten persona labels for the table
PERSONA_ABBR = {
    "Prospective Student": "Student",
    "Parent of Prospective Student": "Parent",
}
def abbreviate_personas(p):
    return " + ".join(PERSONA_ABBR.get(x.strip(), x.strip()) for x in p.split(","))

intent_labels = {
    "informational": "Informational (10)",
    "comparison": "Comparison (8)",
    "recommendation": "Recommendation (8)",
}

grouped = {"informational": [], "comparison": [], "recommendation": []}
for topic, prompt, tag, region, lang, platforms, personas, analysis in prompt_rows:
    grouped[tag].append((topic, prompt, personas))

table_style = TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.white),
    ('TEXTCOLOR', (0,0), (-1,0), colors.black),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('GRID', (0,0), (-1,-1), 0.5, colors.black),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ('LEFTPADDING', (0,0), (-1,-1), 5),
    ('RIGHTPADDING', (0,0), (-1,-1), 5),
])

tiny = ParagraphStyle('Tiny', parent=styles['BodyText'], fontSize=8, leading=10, textColor=colors.black)

for tag in ["informational", "comparison", "recommendation"]:
    story.append(Paragraph(intent_labels[tag], h2))
    table_data = [["Topic", "Prompt", "Personas"]]
    for topic, prompt, personas in grouped[tag]:
        table_data.append([topic, prompt, abbreviate_personas(personas)])
    rows_fmt = [
        [Paragraph(f"<b>{c}</b>", tiny) for c in table_data[0]]
    ] + [
        [Paragraph(r[0], tiny), Paragraph(r[1], tiny), Paragraph(r[2], tiny)]
        for r in table_data[1:]
    ]
    t = Table(rows_fmt, colWidths=[1.7*inch, 3.6*inch, 1.1*inch], repeatRows=1)
    t.setStyle(table_style)
    story.append(t)
    story.append(Spacer(1, 8))

doc.build(story)
print("PDF built")
