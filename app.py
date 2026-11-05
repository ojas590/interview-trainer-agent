"""
Flask API — Interview Trainer Agent Backend
Provides REST endpoints for the frontend to interact with the agent.
"""
import os
import logging
import json
from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, request, jsonify, send_from_directory, session, Response
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__, static_folder="static", static_url_path="")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-me")
CORS(app, supports_credentials=True)

# Lazy-loaded agent
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        from src.agent import get_agent as _get_agent
        _agent = _get_agent()
    return _agent


# ──────────────────────────────────────────────
# Health & Status
# ──────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "Interview Trainer Agent",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "watsonx_configured": bool(
            os.getenv("WATSONX_API_KEY") and
            os.getenv("WATSONX_PROJECT_ID") and
            os.getenv("WATSONX_API_KEY") != "your_ibm_cloud_api_key_here"
        ),
    })


# ──────────────────────────────────────────────
# Session Management
# ──────────────────────────────────────────────

@app.route("/api/session/start", methods=["POST"])
def start_session():
    """
    Start a new interview prep session.
    Body: { name, job_title, experience_level, skills[], resume_text?, target_company? }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    required = ["name", "job_title", "experience_level"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    valid_levels = ["junior", "mid", "senior"]
    if data["experience_level"].lower() not in valid_levels:
        return jsonify({"error": f"experience_level must be one of: {valid_levels}"}), 400

    # Store profile in session
    session["profile"] = {
        "name": data["name"].strip(),
        "job_title": data["job_title"].strip(),
        "experience_level": data["experience_level"].lower(),
        "skills": [s.strip() for s in data.get("skills", []) if s.strip()],
        "resume_text": data.get("resume_text", ""),
        "target_company": data.get("target_company", ""),
        "focus_areas": data.get("focus_areas", ["technical", "behavioral", "hr"]),
    }
    session["started_at"] = datetime.now(timezone.utc).isoformat()

    return jsonify({
        "message": f"Session started for {data['name']}",
        "profile": session["profile"],
    })


@app.route("/api/session/profile", methods=["GET"])
def get_profile():
    """Get current session profile."""
    if "profile" not in session:
        return jsonify({"error": "No active session"}), 404
    return jsonify({"profile": session["profile"]})


# ──────────────────────────────────────────────
# Question Generation
# ──────────────────────────────────────────────

@app.route("/api/questions/generate", methods=["POST"])
def generate_questions():
    """
    Generate personalized interview questions.
    Body: { profile: {...}, num_questions?: int, categories?: [], difficulty?: str }
    OR uses session profile if body.profile is omitted.
    """
    data = request.get_json() or {}
    profile_data = data.get("profile") or session.get("profile")

    if not profile_data:
        return jsonify({"error": "Profile required. Start a session or provide profile in body."}), 400

    num_questions = min(int(data.get("num_questions", 10)), 20)
    categories = data.get("categories", ["technical", "behavioral", "hr"])
    difficulty = data.get("difficulty", None)
    if difficulty and difficulty.lower() not in ("easy", "medium", "hard"):
        difficulty = None

    from src.agent import UserProfile
    profile = UserProfile(
        name=profile_data["name"],
        job_title=profile_data["job_title"],
        experience_level=profile_data["experience_level"],
        skills=profile_data.get("skills", []),
        resume_text=profile_data.get("resume_text", ""),
        target_company=profile_data.get("target_company", ""),
        focus_areas=profile_data.get("focus_areas", categories),
    )

    try:
        agent = get_agent()
        questions = agent.generate_question_set(
            profile, num_questions, categories, difficulty=difficulty
        )
        from dataclasses import asdict
        return jsonify({
            "questions": [asdict(q) for q in questions],
            "count": len(questions),
            "profile_summary": f"{profile.name} | {profile.job_title} | {profile.experience_level}",
        })
    except Exception as e:
        logger.exception("Question generation failed")
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# Full Session (Questions + Strategy + Tips)
# ──────────────────────────────────────────────

@app.route("/api/session/full", methods=["POST"])
def full_session():
    """
    Run a complete interview prep session.
    Returns questions, strategy, confidence boosters, and focus breakdown.
    Body: { profile: {...}, num_questions?: int }
    """
    data = request.get_json() or {}
    profile_data = data.get("profile") or session.get("profile")

    if not profile_data:
        return jsonify({"error": "Profile required"}), 400

    num_questions = min(int(data.get("num_questions", 10)), 20)

    from src.agent import UserProfile
    profile = UserProfile(
        name=profile_data["name"],
        job_title=profile_data["job_title"],
        experience_level=profile_data["experience_level"],
        skills=profile_data.get("skills", []),
        resume_text=profile_data.get("resume_text", ""),
        target_company=profile_data.get("target_company", ""),
    )

    try:
        agent = get_agent()
        interview_session = agent.run_full_session(profile, num_questions)
        return jsonify(agent.to_dict(interview_session))
    except Exception as e:
        logger.exception("Full session failed")
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# Answer Evaluation
# ──────────────────────────────────────────────

@app.route("/api/evaluate", methods=["POST"])
def evaluate_answer():
    """
    Evaluate a user's answer to an interview question.
    Body: { question, answer, profile?: {...} }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    question = data.get("question", "").strip()
    user_answer = data.get("answer", "").strip()

    if not question or not user_answer:
        return jsonify({"error": "Both 'question' and 'answer' are required"}), 400

    if len(user_answer) < 20:
        return jsonify({"error": "Answer too short (minimum 20 characters)"}), 400

    profile_data = data.get("profile") or session.get("profile") or {
        "name": "Candidate",
        "job_title": "Professional",
        "experience_level": "mid",
        "skills": [],
    }

    from src.agent import UserProfile
    profile = UserProfile(
        name=profile_data.get("name", "Candidate"),
        job_title=profile_data.get("job_title", "Professional"),
        experience_level=profile_data.get("experience_level", "mid"),
        skills=profile_data.get("skills", []),
    )

    try:
        agent = get_agent()
        evaluation = agent.evaluate_answer(question, user_answer, profile)
        return jsonify(evaluation)
    except Exception as e:
        logger.exception("Answer evaluation failed")
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# Preparation Strategy
# ──────────────────────────────────────────────

@app.route("/api/strategy", methods=["POST"])
def get_strategy():
    """
    Get a personalized preparation strategy.
    Body: { profile?: {...} }
    """
    data = request.get_json() or {}
    profile_data = data.get("profile") or session.get("profile")

    if not profile_data:
        return jsonify({"error": "Profile required"}), 400

    from src.agent import UserProfile
    profile = UserProfile(
        name=profile_data["name"],
        job_title=profile_data["job_title"],
        experience_level=profile_data["experience_level"],
        skills=profile_data.get("skills", []),
        target_company=profile_data.get("target_company", ""),
    )

    try:
        agent = get_agent()
        strategy = agent.generate_preparation_strategy(profile)
        boosters = agent.generate_confidence_boosters(profile)
        return jsonify({
            "strategy": strategy,
            "confidence_boosters": boosters,
        })
    except Exception as e:
        logger.exception("Strategy generation failed")
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# Feature 1 — Resume Analyzer
# ──────────────────────────────────────────────

@app.route("/api/resume/analyze", methods=["POST"])
def analyze_resume():
    """
    Analyze a resume (PDF file upload or plain text).
    Accepts multipart/form-data with:
      - file: PDF file (optional)
      - text: plain-text resume (optional fallback)
      - job_title: target role
    Returns: { extracted_skills, strengths, missing_skills, overall_assessment, resume_text }
    """
    job_title = request.form.get("job_title", "").strip()
    if not job_title:
        # Try JSON body as well
        body = request.get_json(silent=True) or {}
        job_title = body.get("job_title", "").strip()

    if not job_title:
        return jsonify({"error": "job_title is required"}), 400

    resume_text = ""
    agent = get_agent()

    # Try PDF upload first
    if "file" in request.files:
        file = request.files["file"]
        if file.filename and file.filename.lower().endswith(".pdf"):
            pdf_bytes = file.read()
            if len(pdf_bytes) > 10 * 1024 * 1024:   # 10 MB limit
                return jsonify({"error": "PDF file too large (max 10 MB)"}), 400
            resume_text = agent.extract_text_from_pdf(pdf_bytes)
            if not resume_text:
                return jsonify({"error": "Could not extract text from PDF. Try pasting the text instead."}), 422
        else:
            return jsonify({"error": "Only PDF files are supported"}), 400
    else:
        # Fall back to posted text (JSON or form)
        body = request.get_json(silent=True) or {}
        resume_text = (
            request.form.get("text", "")
            or body.get("resume_text", "")
            or body.get("text", "")
        ).strip()
        if not resume_text:
            return jsonify({"error": "Provide either a PDF file or resume text"}), 400

    try:
        from dataclasses import asdict
        result = agent.analyze_resume(resume_text, job_title)
        return jsonify(asdict(result))
    except Exception as e:
        logger.exception("Resume analysis failed")
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# Feature 3 — Interview History
# ──────────────────────────────────────────────

@app.route("/api/history", methods=["GET"])
def list_history():
    """Return all stored interview sessions, newest first."""
    from src.history_store import get_all_sessions
    return jsonify({"sessions": get_all_sessions()})


@app.route("/api/history", methods=["POST"])
def save_history():
    """
    Save a completed practice session to history.
    Body: { candidate_name, job_title, experience_level, questions[], evaluations[] }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400
    if not data.get("candidate_name") or not data.get("job_title"):
        return jsonify({"error": "candidate_name and job_title are required"}), 400

    from src.history_store import save_session
    session_id = save_session(data)
    return jsonify({"message": "Session saved", "session_id": session_id}), 201


@app.route("/api/history/<session_id>", methods=["GET"])
def get_history_session(session_id):
    """Return a single history session by ID."""
    from src.history_store import get_session
    record = get_session(session_id)
    if not record:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(record)


@app.route("/api/history/<session_id>", methods=["DELETE"])
def delete_history_session(session_id):
    """Delete a history session by ID."""
    from src.history_store import delete_session
    if delete_session(session_id):
        return jsonify({"message": "Session deleted"})
    return jsonify({"error": "Session not found"}), 404


@app.route("/api/history/trend", methods=["GET"])
def score_trend():
    """Return score trend data for charting."""
    candidate_name = request.args.get("name", None)
    from src.history_store import get_score_trend
    return jsonify({"trend": get_score_trend(candidate_name)})


# ──────────────────────────────────────────────
# Feature 4 — PDF Report Download
# ──────────────────────────────────────────────

@app.route("/api/report/pdf", methods=["POST"])
def download_pdf_report():
    """
    Generate and stream a PDF performance report.
    Body: {
        candidate_name, job_title, experience_level,
        questions[], evaluations[], preparation_strategy?
    }
    """
    data = request.get_json(silent=True, force=True)
    if not data:
        return jsonify({"error": "Request body required"}), 400
    if not data.get("candidate_name"):
        return jsonify({"error": "candidate_name is required"}), 400

    data.setdefault(
        "generated_at",
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )

    try:
        from src.pdf_report import generate_report_pdf
        pdf_bytes = generate_report_pdf(data)

        safe_name = "".join(
            c for c in data.get("candidate_name", "report") if c.isalnum() or c in " _-"
        ).strip().replace(" ", "_") or "report"
        filename = f"Interview_Report_{safe_name}.pdf"

        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(pdf_bytes)),
            },
        )
    except ImportError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.exception("PDF report generation failed")
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# Knowledge Base Info
# ──────────────────────────────────────────────

@app.route("/api/roles", methods=["GET"])
def get_roles():
    """Return available job roles and experience levels."""
    return jsonify({
        "roles": [
            "Software Engineer",
            "Data Scientist",
            "Product Manager",
            "DevOps Engineer",
            "Machine Learning Engineer",
            "Frontend Developer",
            "Backend Developer",
            "Full Stack Developer",
            "Business Analyst",
            "UX Designer",
        ],
        "experience_levels": [
            {"value": "junior", "label": "Junior (0-2 years)"},
            {"value": "mid", "label": "Mid-Level (3-5 years)"},
            {"value": "senior", "label": "Senior (6+ years)"},
        ],
        "categories": ["technical", "behavioral", "hr"],
    })


# ──────────────────────────────────────────────
# Frontend serving
# ──────────────────────────────────────────────

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """Serve the React/HTML frontend."""
    static_folder = Path(app.static_folder)
    if path and (static_folder / path).exists():
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


# ──────────────────────────────────────────────
# Error handlers
# ──────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    logger.info(f"Starting Interview Trainer Agent on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
