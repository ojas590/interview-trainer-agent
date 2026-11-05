# Interview Trainer Agent — IBM Granite + RAG

<p align="center">
  <strong>AI-Powered Interview Preparation & Coaching Platform</strong>
</p>

<p align="center">
  <a href="https://interview-trainer-agent-dl0j.onrender.com">Live Demo</a> •
  <a href="https://github.com/ojas590/interview-trainer-agent">GitHub Repository</a>
</p>

---

## 🎯 Problem Statement #22

**Interview Trainer Agent** is an AI-powered interview preparation platform developed using IBM Granite and Retrieval-Augmented Generation (RAG).

The platform helps candidates prepare for job interviews through personalized interview questions, AI-based answer evaluation, resume analysis, and structured preparation strategies.

It uses IBM watsonx.ai, IBM Granite LLM, IBM Slate embeddings, and FAISS vector search to deliver role-specific and context-aware interview preparation.

## 🚀 Live Demo

**Try the application:**  
https://interview-trainer-agent-dl0j.onrender.com

> Note: The application is hosted on Render's free tier. The service may take some time to start after a period of inactivity.

## ✨ Key Features

- **Personalized Interview Questions:** Generates role-specific questions based on the candidate's profile, skills, experience level, and selected difficulty.
- **RAG-Powered Question Generation:** Retrieves relevant information from a curated interview knowledge base before generating responses.
- **IBM Granite Integration:** Uses `ibm/granite-3-3-8b-instruct` for AI-powered generation and evaluation.
- **IBM Slate Embeddings:** Uses `ibm/slate-125m-english-rtrvr` for semantic document embeddings.
- **AI Answer Evaluation:** Evaluates candidate answers and provides a score out of 10, strengths, improvement suggestions, and an improved sample answer.
- **Resume Analyzer:** Helps candidates analyze their resumes and identify areas for improvement.
- **Personalized Preparation Strategy:** Generates a structured two-week interview preparation roadmap.
- **Interview History:** Provides access to previous interview practice activities.
- **PDF Reports:** Supports generating downloadable interview preparation reports.
- **Multiple Interview Categories:** Covers technical, behavioral, HR, and soft-skills questions.
- **FAISS Vector Store:** Enables efficient semantic similarity search over the knowledge base.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| AI / LLM | IBM Granite 3.3 8B Instruct |
| Embeddings | IBM Slate 125M English Retriever |
| AI Platform | IBM watsonx.ai |
| RAG Framework | LangChain |
| Vector Database | FAISS |
| Document Processing | PyPDF2, python-docx |
| Deployment | Render |
| Version Control | Git, GitHub |

## 🏗️ System Architecture

```text
              USER
                |
                v
       Candidate Profile
                |
                v
       Flask REST API
                |
                v
          RAG Pipeline
                |
       +--------+---------+
       |                  |
       v                  v
 Knowledge Base      IBM Slate
       |              Embeddings
       |                  |
       +--------+---------+
                |
                v
          FAISS Vector
             Search
                |
                v
        Relevant Documents
                |
                v
         IBM Granite LLM
                |
                v
       AI-Generated Output
                |
       +--------+---------+
       |        |         |
       v        v         v
    Questions Evaluation Strategy
                |
                v
         User Dashboard
```

## 🧠 IBM Technologies & Models

| Component | IBM Technology | Model |
|---|---|---|
| Question Generation | IBM watsonx.ai | `ibm/granite-3-3-8b-instruct` |
| Answer Evaluation | IBM watsonx.ai | `ibm/granite-3-3-8b-instruct` |
| Preparation Strategy | IBM watsonx.ai | `ibm/granite-3-3-8b-instruct` |
| Semantic Embeddings | IBM watsonx.ai | `ibm/slate-125m-english-rtrvr` |

## 🔍 How RAG Works

The application follows a Retrieval-Augmented Generation workflow:

1. **Knowledge Base Loading:** Curated interview preparation documents are loaded into the application.
2. **Document Embedding:** IBM Slate converts documents into semantic vector representations.
3. **Vector Indexing:** FAISS stores the generated embeddings for efficient similarity search.
4. **Information Retrieval:** Relevant documents are retrieved according to the candidate's role, skills, experience, and query.
5. **Prompt Augmentation:** Retrieved information is included as contextual reference material in the Granite prompt.
6. **Response Generation:** IBM Granite generates personalized interview questions, feedback, and preparation strategies.

## 📁 Project Structure

```text
interview-trainer-agent/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── granite_client.py
│   └── rag_pipeline.py
│
├── knowledge_base/
│   └── Interview preparation JSON files
│
├── static/
│   └── index.html
│
└── tests/
    └── Application tests
```

*Note: Generated vector indexes and sensitive environment files are excluded from version control.*

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.12.10
- Git
- IBM Cloud account
- IBM watsonx.ai project
- IBM watsonx.ai API credentials

### 1. Clone the Repository

```bash
git clone https://github.com/ojas590/interview-trainer-agent.git

cd interview-trainer-agent
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory.

```env
WATSONX_API_KEY=your_ibm_api_key
WATSONX_PROJECT_ID=your_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com

GRANITE_MODEL_ID=ibm/granite-3-3-8b-instruct
GRANITE_EMBEDDING_MODEL_ID=ibm/slate-125m-english-rtrvr

FLASK_SECRET_KEY=your_secret_key
FLASK_ENV=development
FLASK_DEBUG=False
```

Configure the remaining application settings according to your environment.

**Important:** Never commit your `.env` file or expose API keys and credentials publicly.

### 5. Run the Application

```bash
python app.py
```

Open the application in your browser:

```text
http://localhost:5000
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Application health check |
| POST | `/api/session/start` | Save candidate profile |
| GET | `/api/session/profile` | Retrieve candidate profile |
| POST | `/api/questions/generate` | Generate interview questions |
| POST | `/api/session/full` | Generate a complete interview session |
| POST | `/api/evaluate` | Evaluate candidate answers |
| POST | `/api/strategy` | Generate preparation strategy |
| GET | `/api/roles` | Retrieve available roles and levels |

## 🔄 Fallback Mode

The application includes fallback functionality for situations where IBM watsonx.ai credentials are unavailable.

Depending on the configured fallback implementation, the application can use local retrieval and alternative embedding mechanisms to support development and testing.

For full IBM Granite and Slate functionality, valid watsonx.ai credentials are required.

## 🌐 Deployment

The application is deployed using **Render Web Services**.

- Hosting Platform: Render
- Runtime: Python 3.12.10
- Application Server: Gunicorn
- Backend Framework: Flask
- Source Code: GitHub

**Live Application:**  
https://interview-trainer-agent-dl0j.onrender.com

## 🔮 Future Scope

- Voice-based mock interview sessions.
- Real-time speech-to-text interview practice.
- Advanced resume-to-job-description matching.
- Multi-language interview preparation.
- Detailed candidate performance analytics.
- Personalized adaptive learning recommendations.
- Expanded industry-specific knowledge bases.

## 👨‍💻 Project Information

| Detail | Information |
|---|---|
| Project Name | Interview Trainer Agent |
| Problem Statement | #22 |
| Domain | Artificial Intelligence & Generative AI |
| Core Technologies | IBM Granite, RAG, IBM Slate, FAISS |
| Backend | Python Flask |
| Deployment | Render |
| Repository | [GitHub](https://github.com/ojas590/interview-trainer-agent) |
| Live Demo | [Open Application](https://interview-trainer-agent-dl0j.onrender.com) |

---

<p align="center">
  <strong>Built with IBM Granite, RAG, and Generative AI.</strong>
</p>
