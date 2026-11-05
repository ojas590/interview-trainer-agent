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
 Knowledge Base      IBM
