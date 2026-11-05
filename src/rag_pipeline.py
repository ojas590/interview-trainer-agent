"""
RAG Pipeline — Document loading, chunking, embedding, and retrieval
using FAISS vector store with IBM watsonx.ai embeddings.
"""
import os
import json
import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "./data/knowledge_base")
TOP_K = int(os.getenv("TOP_K_RESULTS", "5"))


class Document:
    """Simple document container."""

    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(id={self.metadata.get('id')}, role={self.metadata.get('role')})"


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    Loads knowledge base JSON files, embeds them using IBM watsonx.ai,
    stores them in a FAISS index, and retrieves relevant documents at query time.
    """

    def __init__(self, use_granite_embeddings: bool = True):
        self.use_granite_embeddings = use_granite_embeddings
        self.documents: list[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self._granite = None
        self._load_or_build_index()

    def _get_granite(self):
        """Lazy load Granite client."""
        if self._granite is None:
            try:
                from src.granite_client import get_granite_client
                self._granite = get_granite_client()
            except Exception as e:
                logger.warning(f"Granite client unavailable, using fallback embeddings: {e}")
                self.use_granite_embeddings = False
        return self._granite

    def _load_documents(self) -> list[Document]:
        """Load all JSON knowledge base files into Document objects."""
        docs = []
        kb_path = Path(KNOWLEDGE_BASE_PATH)
        if not kb_path.exists():
            logger.warning(f"Knowledge base path not found: {kb_path}")
            return docs

        for json_file in kb_path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    items = json.load(f)

                for item in items:
                    # Build rich text content for embedding
                    if "question" in item:
                        content_parts = [
                            f"Role: {item.get('role', 'All')}",
                            f"Category: {item.get('category', 'general')}",
                            f"Level: {item.get('level', 'all')}",
                            f"Question: {item.get('question', '')}",
                            f"Answer: {item.get('model_answer', '')}",
                            f"Tips: {item.get('tips', '')}",
                        ]
                        if "keywords" in item:
                            content_parts.append(f"Keywords: {', '.join(item['keywords'])}")
                    else:
                        # Industry insight format
                        content_parts = [
                            f"Topic: {item.get('topic', '')}",
                            f"Role: {item.get('role', 'All')}",
                            f"Content: {item.get('content', '')}",
                        ]

                    content = "\n".join(content_parts)
                    docs.append(Document(content=content, metadata=item))

            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")

        logger.info(f"Loaded {len(docs)} documents from knowledge base")
        return docs

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts using IBM Granite or TF-IDF fallback."""
        if self.use_granite_embeddings:
            try:
                granite = self._get_granite()
                if granite:
                    vectors = granite.get_embeddings(texts)
                    return np.array(vectors, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Granite embedding failed, using TF-IDF fallback: {e}")

        # TF-IDF fallback for local development without credentials
        return self._tfidf_embed(texts)

    def _tfidf_embed(self, texts: list[str]) -> np.ndarray:
        """
        Pure-numpy bag-of-words TF-IDF fallback.
        No scipy/sklearn required — safe on any platform.
        """
        import re
        import math

        STOP = {"a","an","the","is","are","was","were","be","been","of","in",
                "to","for","on","at","by","with","that","this","it","and","or","not"}

        def tokenize(text):
            return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP]

        if not hasattr(self, '_vocab') or self._vocab is None:
            # Build vocabulary from all texts
            from collections import Counter
            all_tokens = [tok for t in texts for tok in tokenize(t)]
            freq = Counter(all_tokens)
            # Keep top 512 most common tokens
            vocab_words = [w for w, _ in freq.most_common(512)]
            self._vocab = {w: i for i, w in enumerate(vocab_words)}

        vocab = self._vocab
        V = len(vocab)
        N = len(texts)
        matrix = np.zeros((N, V), dtype=np.float32)

        for i, text in enumerate(texts):
            tokens = tokenize(text)
            if not tokens:
                continue
            from collections import Counter
            tf = Counter(tokens)
            for tok, count in tf.items():
                if tok in vocab:
                    matrix[i, vocab[tok]] = count / len(tokens)

        # IDF
        df = (matrix > 0).sum(axis=0) + 1
        idf = np.log((N + 1) / df) + 1
        matrix = matrix * idf

        return matrix

    def _build_index(self):
        """Build FAISS index from knowledge base documents."""
        try:
            import faiss
        except ImportError:
            logger.error("faiss-cpu not installed. Run: pip install faiss-cpu")
            raise

        self.documents = self._load_documents()
        if not self.documents:
            raise RuntimeError("No documents loaded from knowledge base")

        texts = [doc.content for doc in self.documents]
        self.embeddings = self._embed_texts(texts)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vecs)

        # Normalize for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = self.embeddings / norms

        self.index.add(normalized)
        logger.info(f"Built FAISS index with {len(self.documents)} documents, dim={dim}")

        # Cache to disk
        self._save_index()

    def _save_index(self):
        """Save FAISS index and documents to disk."""
        try:
            import faiss
            Path(VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, f"{VECTOR_STORE_PATH}/index.faiss")
            with open(f"{VECTOR_STORE_PATH}/documents.pkl", "wb") as f:
                pickle.dump(self.documents, f)
            if hasattr(self, '_vocab') and self._vocab:
                with open(f"{VECTOR_STORE_PATH}/vocab.pkl", "wb") as f:
                    pickle.dump(self._vocab, f)
            logger.info("Saved vector store to disk")
        except Exception as e:
            logger.warning(f"Could not save index to disk: {e}")

    def _load_index(self) -> bool:
        """Load FAISS index from disk if available."""
        try:
            import faiss
            index_path = Path(f"{VECTOR_STORE_PATH}/index.faiss")
            docs_path = Path(f"{VECTOR_STORE_PATH}/documents.pkl")

            if not (index_path.exists() and docs_path.exists()):
                return False

            self.index = faiss.read_index(str(index_path))
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)

            vocab_path = Path(f"{VECTOR_STORE_PATH}/vocab.pkl")
            if vocab_path.exists():
                with open(vocab_path, "rb") as f:
                    self._vocab = pickle.load(f)

            logger.info(f"Loaded vector store: {len(self.documents)} documents")
            return True
        except Exception as e:
            logger.warning(f"Could not load cached index: {e}")
            return False

    def _load_or_build_index(self):
        """Load cached index or build fresh."""
        self._vocab = None
        if not self._load_index():
            self._build_index()

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        role_filter: str = None,
        category_filter: str = None,
        level_filter: str = None,
    ) -> list[Document]:
        """
        Retrieve the most relevant documents for a query.

        Args:
            query: Search query string
            top_k: Number of documents to retrieve
            role_filter: Filter by job role (e.g., 'Software Engineer')
            category_filter: Filter by category ('technical', 'behavioral', 'hr')
            level_filter: Filter by experience level ('junior', 'mid', 'senior')

        Returns:
            List of most relevant Document objects
        """
        import faiss

        query_vec = self._embed_texts([query])
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        # Retrieve more than needed to allow for post-filtering
        fetch_k = min(top_k * 4, len(self.documents))
        scores, indices = self.index.search(query_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            meta = doc.metadata

            # Apply filters
            if role_filter and role_filter.lower() != "all":
                doc_role = meta.get("role", "All").lower()
                if doc_role != "all" and role_filter.lower() not in doc_role:
                    continue

            if category_filter:
                if meta.get("category", "").lower() != category_filter.lower():
                    continue

            if level_filter and level_filter.lower() != "all":
                doc_level = meta.get("level", "all").lower()
                if doc_level != "all" and doc_level != level_filter.lower():
                    continue

            results.append((score, doc))
            if len(results) >= top_k:
                break

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in results]

    def rebuild_index(self):
        """Force rebuild the vector index (useful after adding new documents)."""
        logger.info("Rebuilding vector index...")
        self._tfidf_vectorizer = None
        self._build_index()


# Module-level singleton
_rag_pipeline = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create the singleton RAGPipeline."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
