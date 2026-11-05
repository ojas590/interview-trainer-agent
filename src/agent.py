"""
Interview Trainer Agent — Core Logic
Orchestrates RAG retrieval + IBM Granite generation to produce
personalized interview question sets, model answers, and coaching tips.
"""
import logging
import json
import re
import io
from typing import Optional
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    name: str
    job_title: str
    experience_level: str  # junior | mid | senior
    skills: list[str]
    resume_text: Optional[str] = None
    target_company: Optional[str] = None
    focus_areas: Optional[list[str]] = None  # technical | behavioral | hr


@dataclass
class InterviewQuestion:
    question: str
    category: str
    difficulty: str        # easy | medium | hard
    model_answer: str
    improvement_tips: str
    keywords: list[str]


@dataclass
class ResumeAnalysis:
    """Result of analysing a PDF/text resume."""
    extracted_skills: list[str]
    strengths: list[str]
    missing_skills: list[str]
    overall_assessment: str
    resume_text: str        # cleaned plain text extracted from the resume


@dataclass
class InterviewSession:
    user_profile: UserProfile
    questions: list[InterviewQuestion]
    preparation_strategy: str
    confidence_boosters: list[str]
    focus_areas_breakdown: dict


SYSTEM_PROMPT = """You are an expert interview coach and career consultant with 15+ years of experience 
helping candidates succeed in technical and non-technical interviews across all industries.
Your coaching is specific, actionable, and tailored to the individual's background.
You use the STAR method for behavioral questions and provide concrete, quantifiable examples.
Always be encouraging while being honest about areas for improvement."""


class InterviewTrainerAgent:
    """
    The core Interview Trainer Agent.
    Uses RAG to retrieve relevant interview content and IBM Granite to generate
    personalized questions, model answers, and preparation strategies.
    """

    def __init__(self):
        self._rag = None
        self._granite = None

    def _get_rag(self):
        """Lazy load RAG pipeline."""
        if self._rag is None:
            from src.rag_pipeline import get_rag_pipeline
            self._rag = get_rag_pipeline()
        return self._rag

    def _get_granite(self):
        """Lazy load Granite client."""
        if self._granite is None:
            try:
                from src.granite_client import get_granite_client
                self._granite = get_granite_client()
            except Exception as e:
                logger.warning(f"Granite client unavailable: {e}")
        return self._granite

    def _build_profile_context(self, profile: UserProfile) -> str:
        """Build a concise profile summary for prompts."""
        lines = [
            f"Candidate: {profile.name}",
            f"Target Role: {profile.job_title}",
            f"Experience Level: {profile.experience_level.capitalize()}",
            f"Key Skills: {', '.join(profile.skills) if profile.skills else 'Not specified'}",
        ]
        if profile.target_company:
            lines.append(f"Target Company: {profile.target_company}")
        if profile.resume_text:
            # Include condensed resume (first 800 chars to avoid token bloat)
            snippet = profile.resume_text[:800].replace('\n', ' ')
            lines.append(f"Resume Summary: {snippet}...")
        return "\n".join(lines)

    def _retrieve_context(
        self,
        profile: "UserProfile",
        category: str = None,
        categories: list[str] = None,
    ) -> str:
        """
        Retrieve relevant documents from the knowledge base.

        `categories` (list) takes priority over singular `category` (str).
        When a single-category filter is active, only docs from that category
        are included in the context so Granite has better grounding.
        """
        rag = self._get_rag()

        # Resolve effective category filter
        if categories and len(categories) == 1:
            effective_cat = categories[0]
        else:
            effective_cat = category  # may be None for mixed/all

        cat_query = effective_cat or (
            " ".join(categories) if categories else ""
        )
        query = (
            f"{profile.job_title} {profile.experience_level} "
            f"interview questions {cat_query}".strip()
        )

        docs = rag.retrieve(
            query=query,
            top_k=8,
            role_filter=profile.job_title,
            category_filter=effective_cat,
            level_filter=profile.experience_level,
        )

        if not docs:
            # Broaden search without filters
            docs = rag.retrieve(query=query, top_k=8)

        context_parts = []
        for i, doc in enumerate(docs[:6], 1):
            context_parts.append(f"[Reference {i}]\n{doc.content}")

        return "\n\n".join(context_parts)

    def generate_question_set(
        self,
        profile: UserProfile,
        num_questions: int = 10,
        categories: list[str] = None,
        difficulty: str = None,          # None = mixed, "easy" | "medium" | "hard"
    ) -> list[InterviewQuestion]:
        """
        Generate a personalized set of interview questions with model answers.

        Args:
            profile: User's profile information
            num_questions: Total number of questions to generate
            categories: List of categories to include (technical/behavioral/hr)
            difficulty: Optional difficulty filter ("easy", "medium", "hard")

        Returns:
            List of InterviewQuestion objects of exactly num_questions length
            (or as many as possible without duplication).
        """
        if categories is None:
            categories = ["technical", "behavioral", "hr"]

        profile_ctx = self._build_profile_context(profile)
        # Pass categories into context retrieval so reference material is relevant
        retrieved_ctx = self._retrieve_context(profile, categories=categories)

        # Build category constraint line based on what was requested
        if len(categories) == 1:
            cat_instruction = (
                f"ALL {num_questions} questions MUST be {categories[0]} questions. "
                f"Do NOT include any other category."
            )
            valid_cats = categories[0]
        else:
            cat_instruction = (
                f"Distribute the {num_questions} questions across these categories: "
                f"{', '.join(categories)}. Each category must appear at least once."
            )
            valid_cats = '|'.join(categories)

        # Difficulty constraint for Granite prompt
        if difficulty and difficulty.lower() in ("easy", "medium", "hard"):
            diff_lower = difficulty.lower()
            diff_instruction = (
                f"ALL questions MUST have difficulty level: {diff_lower}. "
                f"Set \"difficulty\": \"{diff_lower}\" for every object."
            )
        else:
            diff_lower = None
            diff_instruction = "Use a mix of easy, medium, and hard difficulty levels."

        prompt = f"""{SYSTEM_PROMPT}

## Candidate Profile
{profile_ctx}

## Reference Interview Materials
{retrieved_ctx}

## Your Task
Generate EXACTLY {num_questions} tailored interview questions for this candidate.
{cat_instruction}
{diff_instruction}

For each question, respond in this EXACT JSON format (array of objects):
[
  {{
    "question": "The interview question",
    "category": "{valid_cats}",
    "difficulty": "easy|medium|hard",
    "model_answer": "A strong, specific model answer using STAR method for behavioral or technical explanation for technical questions",
    "improvement_tips": "2-3 specific tips to improve the answer based on this candidate's background",
    "keywords": ["key1", "key2", "key3"]
  }}
]

Requirements:
1. Exactly {num_questions} objects in the JSON array — no more, no fewer.
2. Specifically tailored to {profile.job_title} at {profile.experience_level} level.
3. Relevant to skills: {', '.join(profile.skills[:5]) if profile.skills else 'general'}.
4. No duplicate questions.
5. Difficulty: {diff_lower or 'mixed (easy → medium → hard)'}.

Return ONLY the JSON array, no other text."""

        granite = self._get_granite()
        if granite:
            try:
                max_tokens = max(3000, num_questions * 350)
                response = granite.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.6,
                )
                questions = self._parse_questions_json(response)
                if questions:
                    questions = self._enforce_category_filter(questions, categories)
                    # Enforce difficulty on Granite output
                    if diff_lower:
                        for q in questions:
                            q.difficulty = diff_lower
                    if len(questions) < num_questions:
                        questions = self._top_up_questions(
                            questions, profile, num_questions, categories,
                            difficulty=diff_lower,
                        )
                    return questions[:num_questions]
            except Exception as e:
                logger.error(f"Granite generation failed: {e}")

        # Fallback: return curated questions from knowledge base
        return self._fallback_questions(profile, num_questions, categories,
                                        difficulty=diff_lower)

    def _enforce_category_filter(
        self, questions: list, categories: list[str]
    ) -> list:
        """Remove questions whose category is not in the requested list."""
        if not categories or set(categories) == {"technical", "behavioral", "hr"}:
            return questions
        cats_lower = [c.lower() for c in categories]
        return [q for q in questions if q.category.lower() in cats_lower]

    def _top_up_questions(
        self,
        existing: list,
        profile: "UserProfile",
        num_questions: int,
        categories: list[str],
        difficulty: str = None,
    ) -> list:
        """
        Fill the gap between existing questions and num_questions using KB fallback.
        Avoids duplicating questions already in `existing`.
        """
        needed = num_questions - len(existing)
        if needed <= 0:
            return existing

        existing_texts = {q.question.strip().lower() for q in existing}
        candidates = self._fallback_questions(
            profile, needed * 3, categories, difficulty=difficulty
        )
        additions = []
        for q in candidates:
            if q.question.strip().lower() not in existing_texts:
                additions.append(q)
                existing_texts.add(q.question.strip().lower())
            if len(additions) >= needed:
                break
        return existing + additions

    def _parse_questions_json(self, response: str) -> list[InterviewQuestion]:
        """Parse JSON response from Granite into InterviewQuestion objects."""
        # Extract JSON array from response (handle markdown code blocks)
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            logger.warning("No JSON array found in Granite response")
            return []

        try:
            data = json.loads(json_match.group())
            questions = []
            for item in data:
                questions.append(InterviewQuestion(
                    question=item.get("question", ""),
                    category=item.get("category", "general"),
                    difficulty=item.get("difficulty", "medium"),
                    model_answer=item.get("model_answer", ""),
                    improvement_tips=item.get("improvement_tips", ""),
                    keywords=item.get("keywords", []),
                ))
            return questions
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}\nResponse: {response[:500]}")
            return []

    def _fallback_questions(
        self,
        profile: "UserProfile",
        num_questions: int,
        categories: list[str] = None,
        difficulty: str = None,
    ) -> list[InterviewQuestion]:
        """
        Return questions from the knowledge base when Granite is unavailable.
        Respects category and difficulty filters and attempts to meet num_questions
        by querying across multiple strategies if the primary search returns too few.
        """
        if categories is None:
            categories = ["technical", "behavioral", "hr"]

        rag = self._get_rag()
        seen_ids: set[str] = set()
        questions: list[InterviewQuestion] = []

        cats_lower = [c.lower() for c in categories]
        diff_lower = difficulty.lower() if difficulty else None

        # Knowledge-base "level" field maps to difficulty roughly
        # junior → easy, mid → medium, senior → hard
        diff_level_map = {"easy": "junior", "medium": "mid", "hard": "senior"}
        kb_level = diff_level_map.get(diff_lower) if diff_lower else None

        def _docs_to_questions(docs) -> list[InterviewQuestion]:
            result = []
            for doc in docs:
                meta = doc.metadata
                if "question" not in meta:
                    continue
                doc_id = meta.get("id", meta.get("question", ""))
                if doc_id in seen_ids:
                    continue
                # Enforce category filter
                doc_cat = meta.get("category", "general").lower()
                if cats_lower and doc_cat not in cats_lower:
                    continue
                seen_ids.add(doc_id)
                # Map KB level to difficulty label; if difficulty filter set,
                # stamp the requested difficulty rather than the raw KB level value
                raw_level = meta.get("level", "medium")
                level_to_diff = {"junior": "easy", "mid": "medium", "senior": "hard",
                                  "all": "medium", "easy": "easy",
                                  "medium": "medium", "hard": "hard"}
                mapped_diff = level_to_diff.get(raw_level.lower(), "medium")
                final_diff = diff_lower if diff_lower else mapped_diff
                result.append(InterviewQuestion(
                    question=meta.get("question", ""),
                    category=meta.get("category", "general"),
                    difficulty=final_diff,
                    model_answer=meta.get("model_answer", ""),
                    improvement_tips=meta.get("tips", ""),
                    keywords=meta.get("keywords", []),
                ))
            return result

        # Pass 1: role-specific retrieval (no level filter — maximises results)
        fetch_k = max(num_questions * 6, 50)
        docs = rag.retrieve(
            query=f"{profile.job_title} {' '.join(cats_lower)} interview questions",
            top_k=fetch_k,
            role_filter=profile.job_title,
        )
        questions.extend(_docs_to_questions(docs))

        # Pass 2: if still short, broaden to all roles
        if len(questions) < num_questions:
            docs = rag.retrieve(
                query=f"interview questions {' '.join(cats_lower)}",
                top_k=fetch_k,
            )
            questions.extend(_docs_to_questions(docs))

        # Pass 3: per-category targeted retrieval to hit the quota
        if len(questions) < num_questions:
            for cat in cats_lower:
                if len(questions) >= num_questions:
                    break
                docs = rag.retrieve(
                    query=f"{cat} interview question",
                    top_k=fetch_k,
                    category_filter=cat,
                )
                questions.extend(_docs_to_questions(docs))

        # Deduplicate by question text (safety net)
        seen_text: set[str] = set()
        unique: list[InterviewQuestion] = []
        for q in questions:
            key = q.question.strip().lower()
            if key not in seen_text:
                seen_text.add(key)
                unique.append(q)

        return unique[:num_questions]

    def generate_preparation_strategy(self, profile: UserProfile) -> str:
        """Generate a personalized interview preparation strategy."""
        profile_ctx = self._build_profile_context(profile)
        ctx = self._retrieve_context(profile, category=None)

        prompt = f"""{SYSTEM_PROMPT}

## Candidate Profile
{profile_ctx}

## Reference Materials
{ctx}

## Task
Create a focused, actionable 2-week interview preparation strategy for this candidate.
Structure it as:
- Week 1: Foundation (technical skills, knowledge gaps)
- Week 2: Practice & Polish (mock interviews, behavioral stories, company research)
- Daily routine (30-60 min per day)
- Top 3 priority areas based on their profile
- Resources to use (platforms, books, websites)

Be specific to their role ({profile.job_title}) and level ({profile.experience_level}).
Keep it under 400 words. Be motivating and practical."""

        granite = self._get_granite()
        if granite:
            try:
                return granite.generate(prompt=prompt, max_tokens=600, temperature=0.5)
            except Exception as e:
                logger.error(f"Strategy generation failed: {e}")

        return self._default_strategy(profile)

    def _default_strategy(self, profile: UserProfile) -> str:
        return f"""**Your 2-Week Interview Prep Plan for {profile.job_title}**

**Week 1 — Build Your Foundation**
- Days 1-2: Review core {profile.job_title} concepts and brush up on fundamentals
- Days 3-4: Practice 2 LeetCode/technical problems daily (Easy→Medium)
- Days 5-7: Write out 5 STAR stories covering leadership, failure, success, conflict, and teamwork

**Week 2 — Practice & Polish**
- Days 8-9: Research target companies, prepare "Why this company?" answers
- Days 10-11: Do 2 full mock interviews (use Pramp, Interviewing.io, or a friend)
- Days 12-13: Refine answers, prepare questions to ask interviewers
- Day 14: Rest, review notes, lay out outfit, prep tech setup

**Daily Routine (45 min)**
- 15 min: Read one technical article or concept
- 20 min: Practice one question out loud (record yourself)
- 10 min: Review and refine your answers

**Top Priorities for {profile.experience_level.capitalize()} Level**
1. Demonstrate depth of knowledge, not just breadth
2. Quantify all achievements with numbers and impact
3. Show you can handle ambiguity and drive decisions independently

**Recommended Resources**
- Cracking the Coding Interview (technical)
- STAR Stories worksheet (behavioral)
- Glassdoor/Blind for company-specific insights"""

    def evaluate_answer(
        self, question: str, user_answer: str, profile: UserProfile
    ) -> dict:
        """
        Evaluate a user's answer to an interview question and provide feedback.

        Returns a dict with: score (1-10), strengths, improvements, revised_answer
        """
        profile_ctx = self._build_profile_context(profile)

        # Retrieve reference material for this question
        rag = self._get_rag()
        docs = rag.retrieve(query=question, top_k=3, role_filter=profile.job_title)
        ref_content = "\n\n".join([doc.content for doc in docs[:2]])

        prompt = f"""{SYSTEM_PROMPT}

## Candidate Profile
{profile_ctx}

## Reference Material
{ref_content}

## Interview Question
{question}

## Candidate's Answer
{user_answer}

## Your Task
Evaluate this answer and respond in EXACT JSON format:
{{
  "score": <integer 1-10>,
  "strengths": ["strength 1", "strength 2"],
  "improvements": ["improvement 1", "improvement 2", "improvement 3"],
  "revised_answer": "A stronger version of their answer, keeping their voice but improving structure and content",
  "overall_feedback": "2-3 sentence summary of how to improve"
}}

Scoring guide: 1-3=weak, 4-6=adequate, 7-8=good, 9-10=excellent.
Return ONLY the JSON object."""

        granite = self._get_granite()
        if granite:
            try:
                response = granite.generate(prompt=prompt, max_tokens=800, temperature=0.4)
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except Exception as e:
                logger.error(f"Answer evaluation failed: {e}")

        return {
            "score": 5,
            "strengths": ["You provided an answer"],
            "improvements": [
                "Use the STAR method (Situation, Task, Action, Result)",
                "Quantify your impact with specific numbers",
                "Keep it under 2 minutes when spoken",
            ],
            "revised_answer": user_answer,
            "overall_feedback": "Practice structuring your answer with clear examples and measurable outcomes.",
        }

    def generate_confidence_boosters(self, profile: UserProfile) -> list[str]:
        """Generate personalized confidence-building tips."""
        granite = self._get_granite()
        if granite:
            try:
                prompt = f"""{SYSTEM_PROMPT}

Candidate: {profile.name}, applying for {profile.job_title} ({profile.experience_level} level).
Skills: {', '.join(profile.skills[:5]) if profile.skills else 'various'}

Generate 5 specific, personalized confidence-boosting statements for their interview.
These should be based on their actual background and skills, not generic platitudes.
Format as a simple JSON array of strings.
Return ONLY the JSON array."""

                response = granite.generate(prompt=prompt, max_tokens=400, temperature=0.7)
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except Exception as e:
                logger.error(f"Confidence booster generation failed: {e}")

        return [
            f"You have real, valuable experience as a {profile.experience_level} {profile.job_title}.",
            "Every question is an opportunity to share your story — you know it better than anyone.",
            "The interviewer wants you to succeed — they're hoping you're the right candidate.",
            f"Your skills in {', '.join(profile.skills[:2]) if profile.skills else 'your field'} are genuinely valuable.",
            "Preparation beats perfection — you've done the work, now trust yourself.",
        ]

    def run_full_session(
        self,
        profile: UserProfile,
        num_questions: int = 10,
    ) -> InterviewSession:
        """
        Run a complete interview preparation session.

        Returns a full InterviewSession with questions, strategy, and tips.
        """
        logger.info(f"Starting interview session for {profile.name} - {profile.job_title}")

        questions = self.generate_question_set(profile, num_questions)
        strategy = self.generate_preparation_strategy(profile)
        confidence_boosters = self.generate_confidence_boosters(profile)

        # Breakdown by category
        breakdown = {"technical": 0, "behavioral": 0, "hr": 0, "other": 0}
        for q in questions:
            cat = q.category.lower() if q.category.lower() in breakdown else "other"
            breakdown[cat] += 1

        session = InterviewSession(
            user_profile=profile,
            questions=questions,
            preparation_strategy=strategy,
            confidence_boosters=confidence_boosters,
            focus_areas_breakdown=breakdown,
        )
        logger.info(f"Session complete: {len(questions)} questions generated")
        return session

    def to_dict(self, session: InterviewSession) -> dict:
        """Serialize an InterviewSession to a JSON-serializable dict."""
        return {
            "user_profile": asdict(session.user_profile),
            "questions": [asdict(q) for q in session.questions],
            "preparation_strategy": session.preparation_strategy,
            "confidence_boosters": session.confidence_boosters,
            "focus_areas_breakdown": session.focus_areas_breakdown,
            "total_questions": len(session.questions),
        }

    # ──────────────────────────────────────────────────────────────
    # Feature 1 — Resume Analyzer
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def extract_text_from_pdf(pdf_bytes: bytes) -> str:
        """
        Extract plain text from a PDF given as raw bytes.
        Uses PyPDF2 (already in requirements). Falls back gracefully on errors.
        """
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n".join(pages).strip()
        except Exception as e:
            logger.warning(f"PDF extraction failed: {e}")
            return ""

    def analyze_resume(
        self, resume_text: str, job_title: str
    ) -> "ResumeAnalysis":
        """
        Analyse resume text against a target job role using IBM Granite + RAG.

        Returns a ResumeAnalysis with extracted skills, strengths, missing skills,
        and an overall assessment.
        """
        # Retrieve role-specific knowledge to ground Granite's analysis
        rag = self._get_rag()
        docs = rag.retrieve(
            query=f"{job_title} required skills experience qualifications",
            top_k=5,
            role_filter=job_title,
        )
        role_ctx = "\n\n".join(d.content for d in docs[:4]) if docs else ""

        prompt = f"""{SYSTEM_PROMPT}

## Target Job Role
{job_title}

## Role Requirements (from knowledge base)
{role_ctx}

## Candidate Resume
{resume_text[:2000]}

## Your Task
Analyse this resume against the target job role and respond in EXACT JSON format:
{{
  "extracted_skills": ["skill1", "skill2", "skill3"],
  "strengths": ["strength 1 from resume", "strength 2", "strength 3"],
  "missing_skills": ["missing skill 1", "missing skill 2", "missing skill 3"],
  "overall_assessment": "3-4 sentence honest and encouraging assessment of fit for the role"
}}

Be specific and grounded in the resume content. Return ONLY the JSON object."""

        granite = self._get_granite()
        if granite:
            try:
                response = granite.generate(
                    prompt=prompt, max_tokens=700, temperature=0.4
                )
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    return ResumeAnalysis(
                        extracted_skills=data.get("extracted_skills", []),
                        strengths=data.get("strengths", []),
                        missing_skills=data.get("missing_skills", []),
                        overall_assessment=data.get("overall_assessment", ""),
                        resume_text=resume_text,
                    )
            except Exception as e:
                logger.error(f"Resume analysis failed: {e}")

        # Fallback: simple keyword extraction
        return self._fallback_resume_analysis(resume_text, job_title)

    def _fallback_resume_analysis(
        self, resume_text: str, job_title: str
    ) -> "ResumeAnalysis":
        """Simple keyword-based resume analysis when Granite is unavailable."""
        import re as _re
        text_lower = resume_text.lower()

        tech_keywords = [
            "python", "java", "javascript", "typescript", "react", "node",
            "sql", "nosql", "aws", "azure", "gcp", "docker", "kubernetes",
            "machine learning", "deep learning", "tensorflow", "pytorch",
            "api", "rest", "graphql", "microservices", "ci/cd", "agile",
            "scrum", "git", "flask", "django", "fastapi", "spark", "kafka",
        ]
        soft_keywords = [
            "led", "managed", "designed", "built", "improved", "optimized",
            "collaborated", "mentored", "delivered", "achieved",
        ]

        found_skills = [k for k in tech_keywords if k in text_lower]
        found_strengths = [f"Evidence of {k} in your resume" for k in soft_keywords if k in text_lower][:3]

        role_skill_map = {
            "software engineer": ["python", "java", "system design", "algorithms", "testing"],
            "data scientist": ["python", "machine learning", "statistics", "sql", "visualization"],
            "product manager": ["roadmap", "agile", "stakeholders", "metrics", "user research"],
            "devops engineer": ["docker", "kubernetes", "ci/cd", "aws", "terraform"],
        }
        role_lower = job_title.lower()
        expected = next(
            (v for k, v in role_skill_map.items() if k in role_lower),
            ["communication", "problem solving", "teamwork"]
        )
        missing = [s for s in expected if s not in text_lower]

        return ResumeAnalysis(
            extracted_skills=found_skills[:10] or ["Skills not detected — paste more resume text"],
            strengths=found_strengths or ["Resume text analysed", "Experience noted", "Background reviewed"],
            missing_skills=missing[:5] or ["Could not determine — paste full resume text"],
            overall_assessment=(
                f"Your resume shows relevant experience for {job_title}. "
                "Add your IBM Granite credentials to get a detailed AI-powered analysis."
            ),
            resume_text=resume_text,
        )


# Singleton
_agent = None


def get_agent() -> InterviewTrainerAgent:
    global _agent
    if _agent is None:
        _agent = InterviewTrainerAgent()
    return _agent
