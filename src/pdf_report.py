"""
PDF Report Generator — Feature 4
Generates a downloadable PDF performance report using only the standard library
+ fpdf2 (a pure-Python PDF library with no system dependencies).
"""
import logging
import io
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def generate_report_pdf(report_data: dict) -> bytes:
    """
    Build a PDF performance report and return it as bytes.

    report_data shape:
    {
        "candidate_name": str,
        "job_title": str,
        "experience_level": str,
        "generated_at": str (ISO),
        "questions": [{question, category, difficulty, model_answer, improvement_tips}, ...],
        "evaluations": [{question, answer, score, strengths, improvements, overall_feedback}, ...],
        "preparation_strategy": str (markdown),
    }
    """
    try:
        from fpdf import FPDF
    except ImportError:
        logger.error("fpdf2 not installed. Run: pip install fpdf2")
        raise ImportError("fpdf2 required for PDF generation. Run: pip install fpdf2")

    name = report_data.get("candidate_name", "Candidate")
    job_title = report_data.get("job_title", "")
    exp_level = report_data.get("experience_level", "").capitalize()
    generated_at = report_data.get(
        "generated_at",
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
    evaluations = report_data.get("evaluations", [])
    questions = report_data.get("questions", [])
    strategy = report_data.get("preparation_strategy", "")

    # Compute overall stats
    scores = [e.get("score", 0) for e in evaluations if isinstance(e.get("score"), (int, float))]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    max_score = max(scores) if scores else 0

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=20, top=20, right=20)

    # ── Page 1: Cover ──────────────────────────────────────────
    pdf.add_page()

    # Title bar
    pdf.set_fill_color(30, 64, 175)   # IBM-style blue
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_xy(20, 10)
    pdf.cell(170, 12, "Interview Performance Report", align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_xy(20, 26)
    pdf.cell(170, 8, "Powered by IBM Granite & watsonx.ai", align="C")

    # Candidate info block
    pdf.set_text_color(30, 30, 30)
    pdf.set_xy(20, 50)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 8, f"Candidate: {name}")
    pdf.ln(9)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_x(20)
    pdf.cell(0, 7, f"Target Role: {job_title}  |  Level: {exp_level}")
    pdf.ln(7)
    pdf.set_x(20)
    pdf.cell(0, 7, f"Report Generated: {generated_at}")
    pdf.ln(10)

    # Horizontal divider
    pdf.set_draw_color(200, 200, 200)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(8)

    # ── Score summary ──────────────────────────────────────────
    if scores:
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 8, "Overall Performance Summary")
        pdf.ln(9)

        _score_color(pdf, avg_score)
        pdf.set_font("Helvetica", "B", 28)
        pdf.set_x(20)
        pdf.cell(0, 14, f"Average Score: {avg_score}/10")
        pdf.set_text_color(30, 30, 30)
        pdf.ln(14)

        pdf.set_font("Helvetica", "", 11)
        pdf.set_x(20)
        pdf.cell(0, 7, f"Questions Evaluated: {len(scores)}   |   Best Score: {max_score}/10   |   Lowest: {min(scores)}/10")
        pdf.ln(8)

        # Score bar
        bar_x = 20
        bar_y = pdf.get_y()
        bar_w = 170 * (avg_score / 10)
        pdf.set_fill_color(224, 231, 255)
        pdf.rect(bar_x, bar_y, 170, 6, 'F')
        _score_color(pdf, avg_score)
        pdf.rect(bar_x, bar_y, bar_w, 6, 'F')
        pdf.set_text_color(30, 30, 30)
        pdf.ln(10)

    # ── Question list (no answers on cover page) ─────────────
    if questions:
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, f"Interview Questions ({len(questions)} total)")
        pdf.ln(9)

        for i, q in enumerate(questions, 1):
            if pdf.get_y() > 260:
                pdf.add_page()
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 64, 175)
            cat = q.get("category", "general").capitalize()
            diff = q.get("difficulty", "medium").capitalize()
            pdf.set_x(20)
            pdf.cell(0, 6, f"Q{i}. [{cat}] [{diff}]")
            pdf.set_text_color(30, 30, 30)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_x(20)
            pdf.multi_cell(170, 6, _clean(q.get("question", "")))
            pdf.ln(2)

    # ── Pages 2+: Evaluations ─────────────────────────────────
    if evaluations:
        pdf.add_page()
        pdf.set_fill_color(30, 64, 175)
        pdf.rect(0, 0, 210, 18, 'F')
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_xy(20, 5)
        pdf.cell(0, 8, "Detailed Evaluation Feedback")
        pdf.set_text_color(30, 30, 30)
        pdf.set_xy(20, 25)

        for i, ev in enumerate(evaluations, 1):
            if pdf.get_y() > 240:
                pdf.add_page()
                pdf.ln(5)

            score = ev.get("score", 0)
            question = ev.get("question", "")
            answer = ev.get("answer", "")
            overall_feedback = ev.get("overall_feedback", "")
            strengths = ev.get("strengths", [])
            improvements = ev.get("improvements", [])

            # Question header
            pdf.set_x(20)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 64, 175)
            pdf.cell(0, 7, f"Question {i}")
            pdf.set_text_color(30, 30, 30)
            pdf.ln(7)
            pdf.set_x(20)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(170, 6, _clean(question))
            pdf.ln(3)

            # Score pill
            _score_color(pdf, score)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_x(20)
            pdf.cell(40, 8, f"Score: {score}/10", border=1, align="C")
            pdf.set_text_color(30, 30, 30)
            pdf.ln(10)

            # Candidate answer (truncated for readability)
            if answer:
                pdf.set_x(20)
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(100, 100, 100)
                pdf.multi_cell(170, 5, f"Your Answer: {_clean(answer[:300])}{'...' if len(answer) > 300 else ''}")
                pdf.set_text_color(30, 30, 30)
                pdf.ln(3)

            # Overall feedback
            if overall_feedback:
                pdf.set_x(20)
                pdf.set_font("Helvetica", "", 10)
                pdf.multi_cell(170, 6, _clean(overall_feedback))
                pdf.ln(3)

            # Strengths
            if strengths:
                pdf.set_x(20)
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(16, 185, 129)
                pdf.cell(0, 6, "Strengths:")
                pdf.set_text_color(30, 30, 30)
                pdf.set_font("Helvetica", "", 10)
                for s in strengths:
                    pdf.set_x(24)
                    pdf.multi_cell(166, 6, f"+ {_clean(s)}")
                pdf.ln(2)

            # Improvements
            if improvements:
                pdf.set_x(20)
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(245, 158, 11)
                pdf.cell(0, 6, "Areas to Improve:")
                pdf.set_text_color(30, 30, 30)
                pdf.set_font("Helvetica", "", 10)
                for imp in improvements:
                    pdf.set_x(24)
                    pdf.multi_cell(166, 6, f"-> {_clean(imp)}")
                pdf.ln(4)

            # Divider
            pdf.set_draw_color(220, 220, 220)
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.ln(6)

    # ── Preparation Strategy page ─────────────────────────────
    if strategy:
        pdf.add_page()
        pdf.set_fill_color(30, 64, 175)
        pdf.rect(0, 0, 210, 18, 'F')
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_xy(20, 5)
        pdf.cell(0, 8, "Preparation Strategy")
        pdf.set_text_color(30, 30, 30)
        pdf.set_xy(20, 25)
        pdf.set_font("Helvetica", "", 10)

        # Strip markdown symbols for PDF
        import re
        clean_strategy = re.sub(r'\*\*(.+?)\*\*', r'\1', strategy)
        clean_strategy = re.sub(r'\*(.+?)\*', r'\1', clean_strategy)
        clean_strategy = re.sub(r'^#{1,3} ', '', clean_strategy, flags=re.MULTILINE)
        pdf.multi_cell(170, 6, _clean(clean_strategy))

    # ── Footer on all pages ──────────────────────────────────
    _add_footer(pdf, name, job_title)

    return bytes(pdf.output())


def _clean(text: str) -> str:
    """Remove non-latin-1 characters that FPDF core fonts can't handle."""
    return str(text or "").encode("latin-1", errors="replace").decode("latin-1")


def _score_color(pdf, score: float):
    """Set fill + text colour based on score tier."""
    if score >= 7:
        pdf.set_fill_color(16, 185, 129)   # green
        pdf.set_text_color(16, 185, 129)
    elif score >= 4:
        pdf.set_fill_color(245, 158, 11)   # amber
        pdf.set_text_color(245, 158, 11)
    else:
        pdf.set_fill_color(239, 68, 68)    # red
        pdf.set_text_color(239, 68, 68)


def _add_footer(pdf: "FPDF", name: str, job_title: str):
    """Add a footer to every page."""
    # In fpdf2, pdf.pages is a dict; use len() to get page count
    total_pages = len(pdf.pages)
    for page_no in range(1, total_pages + 1):
        pdf.page = page_no
        pdf.set_xy(20, 285)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(85, 5, _clean(f"{name} | {job_title}"), align="L")
        pdf.cell(85, 5, _clean(f"Page {page_no}/{total_pages} | Interview Trainer Agent - IBM Granite"), align="R")
