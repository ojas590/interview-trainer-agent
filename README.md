# Interview Trainer Agent — IBM Granite + RAG

> **Problem Statement #22** — An AI-powered Interview Trainer Agent using Retrieval-Augmented Generation (RAG) and IBM Granite to prepare users for job interviews with tailored questions, model answers, and coaching strategies.

## 🎯 Features

- **RAG-Powered Question Generation** — Retrieves role-specific content from a curated knowledge base, then uses IBM Granite to generate personalized questions
- **IBM Granite Integration** — Uses `ibm/granite-3-3-8b-instruct` for all generation tasks
- **IBM Slate Embeddings** — Uses `ibm/slate-125m-english-rtrvr` for semantic document retrieval
- **Answer Evaluation** — Submit your practice answer and get a score (1-10), strengths, improvements, and a revised example
- **Personalized Prep Strategy** — 2-week day-by-day preparation roadmap tailored to your role and level
- **Confidence Boosters** — Personalized affirmations based on your actual background
- **Full Coverage** — Technical, behavioral, and HR/soft skills questions
- **FAISS Vector Store** — Efficient similarity search over the knowledge base

## 🏗️ Architecture

```
User Profile → RAG Query → FAISS Vector Store → Retrieved Docs
                                                       ↓
                                            IBM Granite Prompt
                                                       ↓
                                    Personalized Questions + Answers
```

**Components:**
- `app.py` — Flask REST API server
- `src/agent.py` — Interview Trainer Agent core logic
- `src/rag_pipeline.py` — Document loading, embedding, FAISS indexing, retrieval
- `src/granite_client.py` — IBM watsonx.ai / Granite LLM wrapper
- `data/knowledge_base/` — Curated JSON interview question databases
- `static/index.html` — Single-page web frontend

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- IBM Cloud account (free tier at [cloud.ibm.com](https://cloud.ibm.com))
- IBM watsonx.ai project

### 2. Get IBM Credentials
1. Sign up at [cloud.ibm.com](https://cloud.ibm.com) (free Lite plan works)
2. Go to **watsonx.ai** → Create a project
3. Generate an **API key** from IAM
4. Copy your **Project ID** from the project settings

### 3. Configure Environment
Edit the `.env` file:
```env
WATSONX_API_KEY=your_actual_api_key_here
WATSONX_PROJECT_ID=your_actual_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
python app.py
```

Open your browser at **http://localhost:5000**

## 📁 Project Structure

```
interview-trainer-agent/
├── app.py                      # Flask API server
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (credentials)
├── src/
│   ├── __init__.py
│   ├── agent.py                # Core Interview Trainer Agent logic
│   ├── granite_client.py       # IBM Granite / watsonx.ai integration
│   └── rag_pipeline.py         # RAG pipeline (FAISS + embeddings)
├── data/
│   ├── knowledge_base/
│   │   ├── software_engineering.json
│   │   ├── data_science.json
│   │   ├── product_management.json
│   │   ├── hr_behavioral.json
│   │   └── industry_insights.json
│   └── vector_store/           # Auto-generated FAISS index cache
└── static/
    └── index.html              # Frontend SPA
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check & watsonx.ai status |
| POST | `/api/session/start` | Save user profile to session |
| GET | `/api/session/profile` | Get current session profile |
| POST | `/api/questions/generate` | Generate personalized questions |
| POST | `/api/session/full` | Full session (questions + strategy) |
| POST | `/api/evaluate` | Evaluate a practice answer |
| POST | `/api/strategy` | Generate prep strategy |
| GET | `/api/roles` | List available roles and levels |

### Example: Generate Questions
```bash
curl -X POST http://localhost:5000/api/questions/generate \
  -H "Content-Type: application/json" \
  -d '{
    "profile": {
      "name": "Alex",
      "job_title": "Software Engineer",
      "experience_level": "mid",
      "skills": ["Python", "React", "AWS"]
    },
    "num_questions": 10
  }'
```

### Example: Evaluate Answer
```bash
curl -X POST http://localhost:5000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Tell me about yourself",
    "answer": "I am a software engineer with 4 years of experience...",
    "profile": {"name": "Alex", "job_title": "Software Engineer", "experience_level": "mid", "skills": []}
  }'
```

## 🧠 IBM Technology Used

| Component | IBM Service | Model |
|-----------|-------------|-------|
| Question Generation | watsonx.ai | `ibm/granite-3-3-8b-instruct` |
| Answer Evaluation | watsonx.ai | `ibm/granite-3-3-8b-instruct` |
| Strategy Generation | watsonx.ai | `ibm/granite-3-3-8b-instruct` |
| Semantic Embeddings | watsonx.ai | `ibm/slate-125m-english-rtrvr` |

## 💡 How RAG Works Here

1. **Index Building**: On first run, all JSON knowledge base files are loaded, each document is converted to text, embedded using IBM Slate, and stored in a FAISS vector index.
2. **Retrieval**: For each user query/profile, a semantic search finds the top-K most relevant documents (filtered by role, level, category).
3. **Augmentation**: Retrieved documents are injected into the IBM Granite prompt as reference material.
4. **Generation**: IBM Granite generates personalized, role-specific content grounded in the retrieved knowledge.

## 📝 Fallback Mode

If IBM watsonx.ai credentials are not configured, the system falls back to:
- **TF-IDF embeddings** (scikit-learn) instead of IBM Slate
- **Direct knowledge base retrieval** instead of Granite generation

This allows testing the UI and RAG pipeline without IBM credentials.

## 🔧 Adding New Knowledge

Add JSON files to `data/knowledge_base/` following this schema:
```json
[
  {
    "id": "unique_id",
    "role": "Job Title",
    "category": "technical|behavioral|hr",
    "level": "junior|mid|senior|all",
    "question": "Interview question text",
    "model_answer": "Strong model answer",
    "tips": "Coaching tips",
    "keywords": ["keyword1", "keyword2"]
  }
]
```

Then restart the server to rebuild the FAISS index (or call `GET /api/rebuild-index`).
