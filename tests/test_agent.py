"""
Tests for the Interview Trainer Agent
Run: python -m pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ─── Shared fixture ───────────────────────────────────────────────────────────

def make_profile(job_title="Software Engineer", level="mid", skills=None):
    from src.agent import UserProfile
    return UserProfile(
        name="Test User",
        job_title=job_title,
        experience_level=level,
        skills=skills or ["Python", "AWS"],
    )


# ─── Test RAG Pipeline ────────────────────────────────────────────────────────

class TestRAGPipeline:
    def test_document_loading(self):
        """Test that knowledge base documents load correctly."""
        from src.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(use_granite_embeddings=False)
        assert len(pipeline.documents) > 0, "Should load documents from knowledge base"

    def test_retrieve_returns_results(self):
        """Test that retrieval returns relevant documents."""
        from src.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(use_granite_embeddings=False)
        results = pipeline.retrieve("software engineer technical interview", top_k=3)
        assert len(results) > 0, "Should return at least one document"

    def test_retrieve_with_role_filter(self):
        """Test role-filtered retrieval."""
        from src.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(use_granite_embeddings=False)
        results = pipeline.retrieve(
            "interview questions",
            top_k=5,
            role_filter="Software Engineer"
        )
        assert len(results) > 0

    def test_retrieve_with_category_filter_technical(self):
        """Category filter 'technical' must only return technical docs."""
        from src.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(use_granite_embeddings=False)
        results = pipeline.retrieve(
            "technical interview question",
            top_k=10,
            category_filter="technical",
        )
        for doc in results:
            assert doc.metadata.get("category", "").lower() == "technical", (
                f"Expected category=technical, got {doc.metadata.get('category')} "
                f"for doc id={doc.metadata.get('id')}"
            )

    def test_retrieve_with_category_filter_behavioral(self):
        """Category filter 'behavioral' must only return behavioral docs."""
        from src.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(use_granite_embeddings=False)
        results = pipeline.retrieve(
            "behavioral interview question",
            top_k=10,
            category_filter="behavioral",
        )
        for doc in results:
            assert doc.metadata.get("category", "").lower() == "behavioral", (
                f"Expected category=behavioral, got {doc.metadata.get('category')}"
            )

    def test_retrieve_with_category_filter_hr(self):
        """Category filter 'hr' must only return hr docs."""
        from src.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(use_granite_embeddings=False)
        results = pipeline.retrieve(
            "hr interview question",
            top_k=10,
            category_filter="hr",
        )
        for doc in results:
            assert doc.metadata.get("category", "").lower() == "hr", (
                f"Expected category=hr, got {doc.metadata.get('category')}"
            )

    def test_document_has_required_fields(self):
        """Test that loaded documents have content and metadata."""
        from src.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(use_granite_embeddings=False)
        for doc in pipeline.documents[:5]:
            assert doc.content, "Document should have content"
            assert isinstance(doc.metadata, dict), "Metadata should be a dict"

    def test_knowledge_base_has_enough_questions(self):
        """Knowledge base must contain at least 20 questions with category metadata."""
        from src.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(use_granite_embeddings=False)
        question_docs = [d for d in pipeline.documents if "question" in d.metadata]
        assert len(question_docs) >= 20, (
            f"Need at least 20 questions for max fallback count, got {len(question_docs)}"
        )


# ─── Test Agent — Bug 1: Question Count ──────────────────────────────────────

class TestQuestionCount:
    """Bug 1: generate_question_set must return exactly num_questions (or all available)."""

    def _agent(self):
        from src.agent import InterviewTrainerAgent
        a = InterviewTrainerAgent()
        a._granite = None  # force fallback mode
        return a

    def test_count_5(self):
        agent = self._agent()
        profile = make_profile()
        qs = agent.generate_question_set(profile, num_questions=5)
        assert len(qs) == 5, f"Expected 5 questions, got {len(qs)}"

    def test_count_10(self):
        """This is the main reported bug — selecting 10 returned only 4."""
        agent = self._agent()
        profile = make_profile()
        qs = agent.generate_question_set(profile, num_questions=10)
        assert len(qs) == 10, f"Expected 10 questions, got {len(qs)}"

    def test_count_15(self):
        agent = self._agent()
        profile = make_profile()
        qs = agent.generate_question_set(profile, num_questions=15)
        assert len(qs) == 15, f"Expected 15 questions, got {len(qs)}"

    def test_count_20(self):
        """Selecting 20 questions returned only 7 — must now return 20."""
        agent = self._agent()
        profile = make_profile()
        qs = agent.generate_question_set(profile, num_questions=20)
        assert len(qs) == 20, f"Expected 20 questions, got {len(qs)}"

    def test_no_duplicate_questions(self):
        """No two questions should be identical in the same session."""
        agent = self._agent()
        profile = make_profile()
        qs = agent.generate_question_set(profile, num_questions=15)
        texts = [q.question.strip().lower() for q in qs]
        assert len(texts) == len(set(texts)), (
            f"Duplicate questions found: {[t for t in texts if texts.count(t) > 1]}"
        )

    def test_fallback_count_respects_num_questions(self):
        """_fallback_questions must return exactly num_questions (or all available)."""
        agent = self._agent()
        profile = make_profile()
        qs = agent._fallback_questions(profile, num_questions=10)
        assert len(qs) == 10, f"Fallback should return 10, got {len(qs)}"

    def test_top_up_fills_shortage(self):
        """_top_up_questions must fill the gap without duplicates."""
        agent = self._agent()
        profile = make_profile()
        from src.agent import InterviewQuestion
        existing = [
            InterviewQuestion(
                question="Explain REST vs GraphQL",
                category="technical",
                difficulty="medium",
                model_answer="REST uses fixed endpoints...",
                improvement_tips="Give examples",
                keywords=["REST"],
            )
        ]
        result = agent._top_up_questions(existing, profile, num_questions=5, categories=["technical", "behavioral", "hr"])
        assert len(result) == 5
        texts = [q.question.strip().lower() for q in result]
        assert len(texts) == len(set(texts)), "No duplicates after top-up"


# ─── Test Agent — Bug 2: Category Filtering ──────────────────────────────────

class TestCategoryFiltering:
    """Bug 2: category filtering must be respected throughout the pipeline."""

    def _agent(self):
        from src.agent import InterviewTrainerAgent
        a = InterviewTrainerAgent()
        a._granite = None
        return a

    def test_technical_only_filter(self):
        """All returned questions must be technical when filter=['technical']."""
        agent = self._agent()
        profile = make_profile()
        qs = agent.generate_question_set(profile, num_questions=5, categories=["technical"])
        assert len(qs) > 0, "Should return some questions"
        for q in qs:
            assert q.category.lower() == "technical", (
                f"Expected technical, got '{q.category}' for: {q.question[:60]}"
            )

    def test_behavioral_only_filter(self):
        """All returned questions must be behavioral when filter=['behavioral']."""
        agent = self._agent()
        profile = make_profile()
        qs = agent.generate_question_set(profile, num_questions=5, categories=["behavioral"])
        assert len(qs) > 0
        for q in qs:
            assert q.category.lower() == "behavioral", (
                f"Expected behavioral, got '{q.category}' for: {q.question[:60]}"
            )

    def test_hr_only_filter(self):
        """All returned questions must be hr when filter=['hr']."""
        agent = self._agent()
        profile = make_profile()
        qs = agent.generate_question_set(profile, num_questions=5, categories=["hr"])
        assert len(qs) > 0
        for q in qs:
            assert q.category.lower() == "hr", (
                f"Expected hr, got '{q.category}' for: {q.question[:60]}"
            )

    def test_all_categories_returns_mixed(self):
        """All categories filter should return questions from multiple categories."""
        agent = self._agent()
        profile = make_profile()
        qs = agent.generate_question_set(
            profile, num_questions=15, categories=["technical", "behavioral", "hr"]
        )
        cats = {q.category.lower() for q in qs}
        assert len(cats) > 1, f"Expected multiple categories, got only: {cats}"

    def test_enforce_category_filter_removes_wrong_category(self):
        """_enforce_category_filter must strip questions not in requested categories."""
        agent = self._agent()
        from src.agent import InterviewQuestion
        mixed = [
            InterviewQuestion("Tech Q", "technical", "easy", "answer", "tip", []),
            InterviewQuestion("Beh Q", "behavioral", "easy", "answer", "tip", []),
            InterviewQuestion("HR Q", "hr", "easy", "answer", "tip", []),
        ]
        filtered = agent._enforce_category_filter(mixed, ["technical"])
        assert len(filtered) == 1
        assert filtered[0].category.lower() == "technical"

    def test_enforce_category_filter_no_op_for_all(self):
        """With all three categories, nothing should be removed."""
        agent = self._agent()
        from src.agent import InterviewQuestion
        mixed = [
            InterviewQuestion("Tech Q", "technical", "easy", "answer", "tip", []),
            InterviewQuestion("Beh Q", "behavioral", "easy", "answer", "tip", []),
            InterviewQuestion("HR Q", "hr", "easy", "answer", "tip", []),
        ]
        filtered = agent._enforce_category_filter(mixed, ["technical", "behavioral", "hr"])
        assert len(filtered) == 3

    def test_fallback_respects_technical_category(self):
        """_fallback_questions must only return technical questions when requested."""
        agent = self._agent()
        profile = make_profile()
        qs = agent._fallback_questions(profile, num_questions=5, categories=["technical"])
        for q in qs:
            assert q.category.lower() == "technical", (
                f"Expected technical, got '{q.category}'"
            )

    def test_fallback_respects_behavioral_category(self):
        """_fallback_questions must only return behavioral questions when requested."""
        agent = self._agent()
        profile = make_profile()
        qs = agent._fallback_questions(profile, num_questions=5, categories=["behavioral"])
        for q in qs:
            assert q.category.lower() == "behavioral", (
                f"Expected behavioral, got '{q.category}'"
            )

    def test_fallback_respects_hr_category(self):
        """_fallback_questions must only return hr questions when requested."""
        agent = self._agent()
        profile = make_profile()
        qs = agent._fallback_questions(profile, num_questions=5, categories=["hr"])
        for q in qs:
            assert q.category.lower() == "hr", (
                f"Expected hr, got '{q.category}'"
            )


# ─── Test Agent — Bug 3: Markdown (backend produces correct text) ─────────────

class TestMarkdownStrategy:
    """
    Bug 3: The backend produces the strategy as a markdown string.
    The frontend must render it — we test the backend produces valid markdown
    and the frontend utility (renderMarkdown) produces proper HTML.
    We test the backend side here; frontend rendering is covered by
    a standalone JS logic test via Python string simulation.
    """

    def test_default_strategy_contains_markdown(self):
        """The fallback strategy should contain markdown formatting."""
        from src.agent import InterviewTrainerAgent
        agent = InterviewTrainerAgent()
        agent._granite = None
        profile = make_profile()
        strategy = agent._default_strategy(profile)
        # Must contain bold markers
        assert "**" in strategy, "Default strategy should contain markdown bold (**)"

    def test_default_strategy_has_content(self):
        """Strategy should be substantial enough to be useful."""
        from src.agent import InterviewTrainerAgent
        agent = InterviewTrainerAgent()
        agent._granite = None
        profile = make_profile()
        strategy = agent._default_strategy(profile)
        assert len(strategy) > 200, "Strategy should be detailed"
        assert profile.job_title in strategy

    def test_strategy_contains_week_sections(self):
        """Strategy should mention Week 1 and Week 2."""
        from src.agent import InterviewTrainerAgent
        agent = InterviewTrainerAgent()
        agent._granite = None
        profile = make_profile()
        strategy = agent._default_strategy(profile)
        assert "Week 1" in strategy
        assert "Week 2" in strategy


# ─── Test Agent — existing tests ──────────────────────────────────────────────

class TestInterviewTrainerAgent:
    def _make_profile(self):
        return make_profile()

    def test_parse_valid_json(self):
        from src.agent import InterviewTrainerAgent
        agent = InterviewTrainerAgent()
        mock_response = '''
        [
          {
            "question": "What is a REST API?",
            "category": "technical",
            "difficulty": "easy",
            "model_answer": "REST is an architectural style...",
            "improvement_tips": "Give concrete examples",
            "keywords": ["REST", "API", "HTTP"]
          }
        ]
        '''
        questions = agent._parse_questions_json(mock_response)
        assert len(questions) == 1
        assert questions[0].question == "What is a REST API?"
        assert questions[0].category == "technical"

    def test_parse_invalid_json_returns_empty(self):
        from src.agent import InterviewTrainerAgent
        agent = InterviewTrainerAgent()
        questions = agent._parse_questions_json("not json at all")
        assert questions == []

    def test_full_session_without_granite(self):
        from src.agent import InterviewTrainerAgent
        agent = InterviewTrainerAgent()
        agent._granite = None
        profile = self._make_profile()
        session = agent.run_full_session(profile, num_questions=5)
        assert session.user_profile == profile
        assert len(session.questions) == 5
        assert session.preparation_strategy
        assert len(session.confidence_boosters) == 5

    def test_to_dict_serialization(self):
        from src.agent import InterviewTrainerAgent
        agent = InterviewTrainerAgent()
        agent._granite = None
        profile = self._make_profile()
        session = agent.run_full_session(profile, num_questions=3)
        d = agent.to_dict(session)
        assert "user_profile" in d
        assert "questions" in d
        assert "preparation_strategy" in d
        assert "confidence_boosters" in d
        assert "focus_areas_breakdown" in d
        assert isinstance(d["questions"], list)


# ─── Test Flask API ───────────────────────────────────────────────────────────

class TestFlaskAPI:
    @pytest.fixture
    def client(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        flask_app.app.config["SECRET_KEY"] = "test-secret"
        with flask_app.app.test_client() as client:
            yield client

    def test_health_endpoint(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "ok"
        assert "service" in data

    def test_roles_endpoint(self, client):
        res = client.get("/api/roles")
        assert res.status_code == 200
        data = res.get_json()
        assert "roles" in data
        assert len(data["roles"]) > 0
        assert "experience_levels" in data

    def test_session_start_valid(self, client):
        res = client.post("/api/session/start",
            json={
                "name": "Test User",
                "job_title": "Software Engineer",
                "experience_level": "mid",
                "skills": ["Python"],
            }
        )
        assert res.status_code == 200
        data = res.get_json()
        assert "message" in data

    def test_session_start_missing_fields(self, client):
        res = client.post("/api/session/start", json={"name": "Test"})
        assert res.status_code == 400
        data = res.get_json()
        assert "error" in data

    def test_session_start_invalid_level(self, client):
        res = client.post("/api/session/start",
            json={
                "name": "Test",
                "job_title": "Engineer",
                "experience_level": "expert",
            }
        )
        assert res.status_code == 400

    def test_evaluate_answer_missing_fields(self, client):
        res = client.post("/api/evaluate", json={"question": "Tell me about yourself"})
        assert res.status_code == 400

    def test_evaluate_answer_too_short(self, client):
        res = client.post("/api/evaluate",
            json={"question": "Tell me about yourself", "answer": "Hi"}
        )
        assert res.status_code == 400

    def test_questions_generate_without_profile(self, client):
        """Should return 400 when no profile in session or body."""
        res = client.post("/api/questions/generate", json={})
        assert res.status_code == 400

    def test_questions_generate_with_profile_count_10(self, client):
        """Bug 1 API-level: must return exactly 10 questions."""
        res = client.post("/api/questions/generate",
            json={
                "profile": {
                    "name": "Alex",
                    "job_title": "Software Engineer",
                    "experience_level": "junior",
                    "skills": ["Python"],
                },
                "num_questions": 10,
                "categories": ["technical", "behavioral", "hr"],
            }
        )
        assert res.status_code == 200
        data = res.get_json()
        assert "questions" in data
        assert len(data["questions"]) == 10, (
            f"Expected 10 questions from API, got {len(data['questions'])}"
        )

    def test_questions_generate_technical_only(self, client):
        """Bug 2 API-level: technical filter must return only technical questions."""
        res = client.post("/api/questions/generate",
            json={
                "profile": {
                    "name": "Alex",
                    "job_title": "Software Engineer",
                    "experience_level": "mid",
                    "skills": ["Python"],
                },
                "num_questions": 5,
                "categories": ["technical"],
            }
        )
        assert res.status_code == 200
        data = res.get_json()
        assert len(data["questions"]) > 0
        for q in data["questions"]:
            assert q["category"].lower() == "technical", (
                f"Expected technical category, got '{q['category']}'"
            )

    def test_questions_generate_behavioral_only(self, client):
        """Bug 2 API-level: behavioral filter must return only behavioral questions."""
        res = client.post("/api/questions/generate",
            json={
                "profile": {
                    "name": "Alex",
                    "job_title": "Software Engineer",
                    "experience_level": "mid",
                    "skills": ["Python"],
                },
                "num_questions": 5,
                "categories": ["behavioral"],
            }
        )
        assert res.status_code == 200
        data = res.get_json()
        assert len(data["questions"]) > 0
        for q in data["questions"]:
            assert q["category"].lower() == "behavioral", (
                f"Expected behavioral category, got '{q['category']}'"
            )

    def test_questions_generate_hr_only(self, client):
        """Bug 2 API-level: hr filter must return only hr questions."""
        res = client.post("/api/questions/generate",
            json={
                "profile": {
                    "name": "Alex",
                    "job_title": "Software Engineer",
                    "experience_level": "mid",
                    "skills": ["Python"],
                },
                "num_questions": 5,
                "categories": ["hr"],
            }
        )
        assert res.status_code == 200
        data = res.get_json()
        assert len(data["questions"]) > 0
        for q in data["questions"]:
            assert q["category"].lower() == "hr", (
                f"Expected hr category, got '{q['category']}'"
            )

    def test_questions_generate_all_categories_returns_mixed(self, client):
        """All categories should produce a mix."""
        res = client.post("/api/questions/generate",
            json={
                "profile": {
                    "name": "Alex",
                    "job_title": "Software Engineer",
                    "experience_level": "mid",
                    "skills": ["Python"],
                },
                "num_questions": 15,
                "categories": ["technical", "behavioral", "hr"],
            }
        )
        assert res.status_code == 200
        data = res.get_json()
        cats = {q["category"].lower() for q in data["questions"]}
        assert len(cats) > 1, f"Expected mixed categories, got only: {cats}"

    def test_strategy_endpoint_returns_markdown_string(self, client):
        """Bug 3 API-level: strategy must be returned as a string containing markdown."""
        res = client.post("/api/strategy",
            json={
                "profile": {
                    "name": "Alex",
                    "job_title": "Software Engineer",
                    "experience_level": "mid",
                    "skills": ["Python"],
                }
            }
        )
        assert res.status_code == 200
        data = res.get_json()
        assert "strategy" in data
        assert isinstance(data["strategy"], str)
        assert len(data["strategy"]) > 100
        # The strategy should contain markdown formatting (** for bold)
        assert "**" in data["strategy"], (
            "Strategy should contain markdown bold formatting (**)"
        )
        assert "confidence_boosters" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
