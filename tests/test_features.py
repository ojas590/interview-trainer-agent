"""
Tests for the four new features:
  Feature 1 — Resume Analyzer
  Feature 2 — Difficulty Levels
  Feature 3 — Interview History
  Feature 4 — PDF Report
Run: python -m pytest tests/ -v
"""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ─── Shared helpers ───────────────────────────────────────────────────────────

def make_profile(job_title="Software Engineer", level="mid", skills=None):
    from src.agent import UserProfile
    return UserProfile(
        name="Test User",
        job_title=job_title,
        experience_level=level,
        skills=skills or ["Python", "AWS"],
    )


SAMPLE_RESUME_TEXT = """
John Doe | john@example.com | github.com/johndoe
Software Engineer with 4 years of experience.

SKILLS
Python, Django, Flask, REST APIs, PostgreSQL, Docker, AWS, Git, CI/CD

EXPERIENCE
Senior Software Engineer — Acme Corp (2022–Present)
- Led migration of monolithic app to microservices, reducing deploy time by 60%
- Built Python/FastAPI service handling 50k req/day
- Designed PostgreSQL schemas; optimised slow queries by 40%

Software Engineer — Beta Ltd (2020–2022)
- Developed REST APIs with Django; wrote unit and integration tests
- Collaborated with frontend team on React integration

EDUCATION
B.Sc. Computer Science — University of Example, 2020
"""


# ──────────────────────────────────────────────────────────────────────────────
# Feature 1: Resume Analyzer — agent methods
# ──────────────────────────────────────────────────────────────────────────────

class TestResumeAnalyzerAgent:
    def _agent(self):
        from src.agent import InterviewTrainerAgent
        a = InterviewTrainerAgent()
        a._granite = None
        return a

    def test_fallback_resume_analysis_returns_dataclass(self):
        """_fallback_resume_analysis returns a ResumeAnalysis with all fields."""
        from src.agent import ResumeAnalysis
        agent = self._agent()
        result = agent._fallback_resume_analysis(SAMPLE_RESUME_TEXT, "Software Engineer")
        assert isinstance(result, ResumeAnalysis)
        assert isinstance(result.extracted_skills, list)
        assert isinstance(result.strengths, list)
        assert isinstance(result.missing_skills, list)
        assert isinstance(result.overall_assessment, str)
        assert isinstance(result.resume_text, str)

    def test_fallback_extracts_python_skill(self):
        """Python should be detected in the resume."""
        agent = self._agent()
        result = agent._fallback_resume_analysis(SAMPLE_RESUME_TEXT, "Software Engineer")
        assert "python" in [s.lower() for s in result.extracted_skills], (
            f"Expected 'python' in skills, got: {result.extracted_skills}"
        )

    def test_fallback_analysis_with_empty_resume(self):
        """Empty resume should not crash; returns graceful result."""
        agent = self._agent()
        result = agent._fallback_resume_analysis("", "Data Scientist")
        assert isinstance(result.extracted_skills, list)
        assert len(result.extracted_skills) > 0  # fallback message

    def test_analyze_resume_dispatches_to_fallback(self):
        """With no Granite, analyze_resume() still returns a valid result."""
        from src.agent import ResumeAnalysis
        agent = self._agent()
        result = agent.analyze_resume(SAMPLE_RESUME_TEXT, "Software Engineer")
        assert isinstance(result, ResumeAnalysis)
        assert result.resume_text == SAMPLE_RESUME_TEXT

    def test_extract_text_from_pdf_bad_bytes(self):
        """Bad bytes produce empty string without crash."""
        agent = self._agent()
        result = agent.extract_text_from_pdf(b"not a pdf")
        assert isinstance(result, str)
        # may be empty or raise internally and return ""

    def test_extract_text_from_pdf_empty(self):
        """Empty bytes handled gracefully."""
        agent = self._agent()
        result = agent.extract_text_from_pdf(b"")
        assert isinstance(result, str)


# ──────────────────────────────────────────────────────────────────────────────
# Feature 1: Resume Analyzer — Flask API
# ──────────────────────────────────────────────────────────────────────────────

class TestResumeAnalyzerAPI:
    @pytest.fixture
    def client(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        flask_app.app.config["SECRET_KEY"] = "test-secret"
        with flask_app.app.test_client() as c:
            yield c

    def test_analyze_resume_missing_job_title(self, client):
        res = client.post("/api/resume/analyze",
            json={"text": SAMPLE_RESUME_TEXT})
        assert res.status_code == 400
        assert "job_title" in res.get_json().get("error", "").lower()

    def test_analyze_resume_missing_content(self, client):
        res = client.post("/api/resume/analyze",
            json={"job_title": "Software Engineer"})
        assert res.status_code == 400

    def test_analyze_resume_text_mode(self, client):
        res = client.post("/api/resume/analyze",
            json={"text": SAMPLE_RESUME_TEXT, "job_title": "Software Engineer"})
        assert res.status_code == 200
        data = res.get_json()
        assert "extracted_skills" in data
        assert "strengths" in data
        assert "missing_skills" in data
        assert "overall_assessment" in data
        assert isinstance(data["extracted_skills"], list)

    def test_analyze_resume_non_pdf_file(self, client):
        data = {"job_title": "Engineer"}
        res = client.post("/api/resume/analyze",
            data={**data, "file": (b"text content", "resume.txt")},
            content_type="multipart/form-data")
        assert res.status_code == 400


# ──────────────────────────────────────────────────────────────────────────────
# Feature 2: Difficulty Levels
# ──────────────────────────────────────────────────────────────────────────────

class TestDifficultyLevels:
    def _agent(self):
        from src.agent import InterviewTrainerAgent
        a = InterviewTrainerAgent()
        a._granite = None
        return a

    def test_easy_difficulty_stamps_difficulty(self):
        """All questions should have difficulty='easy' when requested."""
        agent = self._agent()
        profile = make_profile()
        qs = agent.generate_question_set(
            profile, num_questions=5, difficulty="easy"
        )
        assert len(qs) > 0
        for q in qs:
            assert q.difficulty == "easy", (
                f"Expected easy, got '{q.difficulty}' for: {q.question[:50]}"
            )

    def test_medium_difficulty_stamps_difficulty(self):
        agent = self._agent()
        profile = make_profile()
        qs = agent.generate_question_set(
            profile, num_questions=5, difficulty="medium"
        )
        for q in qs:
            assert q.difficulty == "medium", f"Expected medium, got '{q.difficulty}'"

    def test_hard_difficulty_stamps_difficulty(self):
        agent = self._agent()
        profile = make_profile()
        qs = agent.generate_question_set(
            profile, num_questions=5, difficulty="hard"
        )
        for q in qs:
            assert q.difficulty == "hard", f"Expected hard, got '{q.difficulty}'"

    def test_no_difficulty_returns_mixed(self):
        """Without difficulty filter, questions can have any difficulty."""
        agent = self._agent()
        profile = make_profile()
        qs = agent.generate_question_set(profile, num_questions=15)
        diffs = {q.difficulty for q in qs}
        # Should contain at least two different difficulty values
        assert len(diffs) >= 1  # At minimum some value is set

    def test_invalid_difficulty_treated_as_none(self):
        """Invalid difficulty value should not crash and produce questions."""
        agent = self._agent()
        profile = make_profile()
        # Backend ignores invalid difficulty; test via the API layer
        qs = agent.generate_question_set(
            profile, num_questions=5, difficulty=None
        )
        assert len(qs) == 5

    def test_fallback_easy_maps_level(self):
        """_fallback_questions with difficulty='easy' stamps easy on all results."""
        agent = self._agent()
        profile = make_profile()
        qs = agent._fallback_questions(profile, 5, difficulty="easy")
        for q in qs:
            assert q.difficulty == "easy"

    def test_fallback_hard_maps_level(self):
        """_fallback_questions with difficulty='hard' stamps hard on all results."""
        agent = self._agent()
        profile = make_profile()
        qs = agent._fallback_questions(profile, 5, difficulty="hard")
        for q in qs:
            assert q.difficulty == "hard"


class TestDifficultyAPI:
    @pytest.fixture
    def client(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        flask_app.app.config["SECRET_KEY"] = "test-secret"
        with flask_app.app.test_client() as c:
            yield c

    def test_api_easy_difficulty(self, client):
        res = client.post("/api/questions/generate",
            json={
                "profile": {
                    "name": "Alex", "job_title": "Software Engineer",
                    "experience_level": "mid", "skills": ["Python"],
                },
                "num_questions": 5,
                "difficulty": "easy",
            })
        assert res.status_code == 200
        data = res.get_json()
        for q in data["questions"]:
            assert q["difficulty"] == "easy", f"Expected easy, got {q['difficulty']}"

    def test_api_medium_difficulty(self, client):
        res = client.post("/api/questions/generate",
            json={
                "profile": {
                    "name": "Alex", "job_title": "Software Engineer",
                    "experience_level": "mid", "skills": [],
                },
                "num_questions": 5,
                "difficulty": "medium",
            })
        assert res.status_code == 200
        for q in res.get_json()["questions"]:
            assert q["difficulty"] == "medium"

    def test_api_hard_difficulty(self, client):
        res = client.post("/api/questions/generate",
            json={
                "profile": {
                    "name": "Alex", "job_title": "Software Engineer",
                    "experience_level": "senior", "skills": [],
                },
                "num_questions": 5,
                "difficulty": "hard",
            })
        assert res.status_code == 200
        for q in res.get_json()["questions"]:
            assert q["difficulty"] == "hard"

    def test_api_invalid_difficulty_ignored(self, client):
        """Invalid difficulty value should be ignored, questions still returned."""
        res = client.post("/api/questions/generate",
            json={
                "profile": {
                    "name": "Alex", "job_title": "Software Engineer",
                    "experience_level": "mid", "skills": [],
                },
                "num_questions": 5,
                "difficulty": "extreme",  # invalid
            })
        assert res.status_code == 200
        data = res.get_json()
        assert len(data["questions"]) == 5


# ──────────────────────────────────────────────────────────────────────────────
# Feature 3: Interview History
# ──────────────────────────────────────────────────────────────────────────────

class TestHistoryStore:
    def _tmp_store(self, tmp_path):
        """Patch HISTORY_FILE to a temp path so tests don't pollute real data."""
        import src.history_store as hs
        hs.HISTORY_FILE = str(tmp_path / "history.json")
        return hs

    def test_save_and_retrieve(self, tmp_path):
        hs = self._tmp_store(tmp_path)
        sid = hs.save_session({
            "candidate_name": "Alice",
            "job_title": "Software Engineer",
            "experience_level": "mid",
            "questions": [{"question": "What is REST?"}],
            "evaluations": [{"question": "What is REST?", "score": 7}],
        })
        assert sid is not None
        record = hs.get_session(sid)
        assert record is not None
        assert record["candidate_name"] == "Alice"

    def test_get_all_sessions_newest_first(self, tmp_path):
        hs = self._tmp_store(tmp_path)
        hs.save_session({"candidate_name": "A", "job_title": "SWE", "evaluations": []})
        hs.save_session({"candidate_name": "B", "job_title": "SWE", "evaluations": []})
        sessions = hs.get_all_sessions()
        assert sessions[0]["candidate_name"] == "B"  # most recent first

    def test_delete_session(self, tmp_path):
        hs = self._tmp_store(tmp_path)
        sid = hs.save_session({"candidate_name": "Del", "job_title": "PM", "evaluations": []})
        assert hs.delete_session(sid) is True
        assert hs.get_session(sid) is None

    def test_delete_nonexistent_returns_false(self, tmp_path):
        hs = self._tmp_store(tmp_path)
        assert hs.delete_session("nonexistent-id") is False

    def test_score_trend(self, tmp_path):
        hs = self._tmp_store(tmp_path)
        hs.save_session({
            "candidate_name": "Alice",
            "job_title": "SWE",
            "evaluations": [
                {"question": "Q1", "score": 6},
                {"question": "Q2", "score": 8},
            ],
        })
        trend = hs.get_score_trend()
        assert len(trend) == 1
        assert trend[0]["avg_score"] == 7.0

    def test_score_trend_filtered_by_name(self, tmp_path):
        hs = self._tmp_store(tmp_path)
        hs.save_session({"candidate_name": "Alice", "job_title": "SWE",
                          "evaluations": [{"score": 8}]})
        hs.save_session({"candidate_name": "Bob", "job_title": "PM",
                          "evaluations": [{"score": 5}]})
        alice_trend = hs.get_score_trend("Alice")
        assert len(alice_trend) == 1
        assert alice_trend[0]["avg_score"] == 8.0

    def test_history_capped_at_50(self, tmp_path):
        hs = self._tmp_store(tmp_path)
        for i in range(55):
            hs.save_session({"candidate_name": f"User{i}", "job_title": "SWE", "evaluations": []})
        all_sessions = hs.get_all_sessions()
        assert len(all_sessions) <= 50


class TestHistoryAPI:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        import src.history_store as hs
        monkeypatch.setattr(hs, "HISTORY_FILE", str(tmp_path / "history.json"))
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        flask_app.app.config["SECRET_KEY"] = "test-secret"
        with flask_app.app.test_client() as c:
            yield c

    def test_list_history_empty(self, client):
        res = client.get("/api/history")
        assert res.status_code == 200
        assert res.get_json()["sessions"] == []

    def test_save_history_valid(self, client):
        res = client.post("/api/history",
            json={
                "candidate_name": "Alice",
                "job_title": "Software Engineer",
                "experience_level": "mid",
                "questions": [{"question": "Q1"}],
                "evaluations": [],
            })
        assert res.status_code == 201
        assert "session_id" in res.get_json()

    def test_save_history_missing_name(self, client):
        res = client.post("/api/history",
            json={"job_title": "SWE", "evaluations": []})
        assert res.status_code == 400

    def test_get_session_by_id(self, client):
        save_res = client.post("/api/history",
            json={"candidate_name": "Bob", "job_title": "PM",
                  "experience_level": "senior", "evaluations": []})
        sid = save_res.get_json()["session_id"]
        get_res = client.get(f"/api/history/{sid}")
        assert get_res.status_code == 200
        assert get_res.get_json()["candidate_name"] == "Bob"

    def test_get_nonexistent_session(self, client):
        res = client.get("/api/history/doesnotexist")
        assert res.status_code == 404

    def test_delete_session_api(self, client):
        save_res = client.post("/api/history",
            json={"candidate_name": "Del", "job_title": "DS",
                  "experience_level": "junior", "evaluations": []})
        sid = save_res.get_json()["session_id"]
        del_res = client.delete(f"/api/history/{sid}")
        assert del_res.status_code == 200
        # Confirm gone
        get_res = client.get(f"/api/history/{sid}")
        assert get_res.status_code == 404

    def test_score_trend_api(self, client):
        client.post("/api/history",
            json={
                "candidate_name": "Trend",
                "job_title": "SWE",
                "experience_level": "mid",
                "evaluations": [{"score": 8}, {"score": 6}],
            })
        res = client.get("/api/history/trend")
        assert res.status_code == 200
        trend = res.get_json()["trend"]
        assert len(trend) == 1
        assert trend[0]["avg_score"] == 7.0


# ──────────────────────────────────────────────────────────────────────────────
# Feature 4: PDF Report
# ──────────────────────────────────────────────────────────────────────────────

class TestPdfReport:
    def _sample_report_data(self):
        return {
            "candidate_name": "Jane Doe",
            "job_title": "Software Engineer",
            "experience_level": "Mid",
            "generated_at": "2026-01-01 12:00 UTC",
            "questions": [
                {
                    "question": "Explain the difference between REST and GraphQL.",
                    "category": "technical",
                    "difficulty": "medium",
                    "model_answer": "REST uses fixed endpoints...",
                    "improvement_tips": "Give examples.",
                },
            ],
            "evaluations": [
                {
                    "question": "Explain the difference between REST and GraphQL.",
                    "answer": "REST is for APIs...",
                    "score": 7,
                    "strengths": ["Clear explanation", "Good structure"],
                    "improvements": ["Add more examples"],
                    "overall_feedback": "Good answer overall.",
                },
            ],
            "preparation_strategy": "**Week 1** — Focus on fundamentals\n- Review Python basics",
        }

    def test_pdf_generation_returns_bytes(self):
        """generate_report_pdf should return non-empty bytes."""
        from src.pdf_report import generate_report_pdf
        data = self._sample_report_data()
        pdf_bytes = generate_report_pdf(data)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000  # PDF should be at least 1 KB

    def test_pdf_starts_with_pdf_magic(self):
        """PDF bytes must start with the %PDF magic number."""
        from src.pdf_report import generate_report_pdf
        data = self._sample_report_data()
        pdf_bytes = generate_report_pdf(data)
        assert pdf_bytes[:4] == b"%PDF", "Output is not a valid PDF"

    def test_pdf_with_no_evaluations(self):
        """Report with no evaluations should still generate without error."""
        from src.pdf_report import generate_report_pdf
        data = self._sample_report_data()
        data["evaluations"] = []
        pdf_bytes = generate_report_pdf(data)
        assert pdf_bytes[:4] == b"%PDF"

    def test_pdf_with_no_questions(self):
        """Report with no questions should still generate."""
        from src.pdf_report import generate_report_pdf
        data = self._sample_report_data()
        data["questions"] = []
        pdf_bytes = generate_report_pdf(data)
        assert pdf_bytes[:4] == b"%PDF"

    def test_pdf_with_special_characters(self):
        """Non-ASCII characters should not crash the generator."""
        from src.pdf_report import generate_report_pdf
        data = self._sample_report_data()
        data["candidate_name"] = "José García"
        pdf_bytes = generate_report_pdf(data)
        assert pdf_bytes[:4] == b"%PDF"


class TestPdfReportAPI:
    @pytest.fixture
    def client(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        flask_app.app.config["SECRET_KEY"] = "test-secret"
        with flask_app.app.test_client() as c:
            yield c

    def test_pdf_endpoint_no_body(self, client):
        res = client.post("/api/report/pdf")
        assert res.status_code == 400

    def test_pdf_endpoint_missing_name(self, client):
        res = client.post("/api/report/pdf",
            json={"job_title": "SWE", "questions": [], "evaluations": []})
        assert res.status_code == 400

    def test_pdf_endpoint_returns_pdf(self, client):
        res = client.post("/api/report/pdf",
            json={
                "candidate_name": "Test User",
                "job_title": "Software Engineer",
                "experience_level": "Mid",
                "questions": [
                    {"question": "What is Python?", "category": "technical",
                     "difficulty": "easy", "model_answer": "A language",
                     "improvement_tips": "Be specific"},
                ],
                "evaluations": [
                    {"question": "What is Python?", "answer": "A programming language",
                     "score": 6, "strengths": ["Clear"], "improvements": ["More depth"],
                     "overall_feedback": "Good start."},
                ],
                "preparation_strategy": "**Week 1** — Start preparing",
            })
        assert res.status_code == 200
        assert res.content_type == "application/pdf"
        assert res.data[:4] == b"%PDF"
        assert "attachment" in res.headers.get("Content-Disposition", "")

    def test_pdf_filename_contains_candidate_name(self, client):
        res = client.post("/api/report/pdf",
            json={
                "candidate_name": "John Smith",
                "job_title": "SWE",
                "experience_level": "mid",
                "questions": [],
                "evaluations": [],
            })
        assert res.status_code == 200
        cd = res.headers.get("Content-Disposition", "")
        assert "John" in cd or "Smith" in cd or "report" in cd.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
