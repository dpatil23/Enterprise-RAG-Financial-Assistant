import os
import json
import pickle
import numpy as np
import faiss
from app.core.config import settings


class VectorStore:
    """
    FAISS-based vector store with JSON metadata persistence.

    Why FAISS over ChromaDB?
    FAISS (Facebook AI Similarity Search) is the gold-standard library
    for fast vector similarity search. It ships pre-built wheels for
    all platforms (no C++ compiler needed on Windows) and is used at
    massive scale in production at Meta, Spotify, and others.

    Architecture:
    - FAISS Index: stores the raw float32 vectors and handles fast search
    - metadata.json: stores the original text chunks and source info
      alongside the index (FAISS only stores vectors, not text)
    - index.faiss: the binary FAISS index file, persisted to disk

    This two-file design (vectors + metadata) is the standard pattern
    for production FAISS deployments.
    """

    def __init__(self):
        self._persist_dir = settings.CHROMA_PERSIST_DIR
        os.makedirs(self._persist_dir, exist_ok=True)
        self._index_path = os.path.join(self._persist_dir, "index.faiss")
        self._meta_path = os.path.join(self._persist_dir, "metadata.json")

        # Load existing index + metadata, or create fresh ones
        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            self._index = faiss.read_index(self._index_path)
            with open(self._meta_path, "r") as f:
                self._metadata = json.load(f)
            print(f"[VectorStore] Loaded existing index with {self._index.ntotal} vectors.")
        else:
            # 384 is the dimensionality of all-MiniLM-L6-v2 embeddings
            # IndexFlatIP = Inner Product (used for cosine similarity on normalized vectors)
            self._index = faiss.IndexFlatIP(384)
            self._metadata = []  # list of {text, source, chunk_index}
            print("[VectorStore] Created fresh FAISS index.")

    def _save(self):
        """Persist the FAISS index and metadata to disk."""
        os.makedirs(self._persist_dir, exist_ok=True)
        faiss.write_index(self._index, self._index_path)
        with open(self._meta_path, "w") as f:
            json.dump(self._metadata, f)

    def add_chunks(
        self,
        collection_name: str,
        chunks: list[str],
        embeddings: list[list[float]],
        doc_id: str,
    ) -> int:
        """
        Normalize and add embedding vectors to the FAISS index.
        Normalization is required to use Inner Product as cosine similarity.
        """
        vectors = np.array(embeddings, dtype=np.float32)
        # L2-normalize so that inner product == cosine similarity
        faiss.normalize_L2(vectors)
        self._index.add(vectors)

        for i, chunk in enumerate(chunks):
            self._metadata.append({
                "text": chunk,
                "source": doc_id,
                "chunk_index": i,
            })

        self._save()
        return len(chunks)

    def query(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict]:
        """
        Find top-K most similar chunks via cosine similarity.
        Returns an empty list if the index has no vectors yet.
        """
        if self._index.ntotal == 0:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        scores, indices = self._index.search(query_vec, min(top_k, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            results.append({
                "text": meta["text"],
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "similarity_score": round(float(score), 4),
            })
        return results

    def list_documents(self, collection_name: str) -> list[str]:
        """Return unique document IDs stored in the index."""
        return list({m["source"] for m in self._metadata})


# Singleton instance
vector_store = VectorStore()
