from sentence_transformers import SentenceTransformer
from app.core.config import settings


class Embedder:
    """
    Wraps the sentence-transformers model.
    
    Why this model?
    'all-MiniLM-L6-v2' is a tiny (80MB) but powerful model that maps
    text to a 384-dimensional vector space. It runs on CPU, is free,
    and achieves 95% of OpenAI's embedding quality for RAG tasks.
    
    The singleton pattern here is CRITICAL: loading the model takes ~2 seconds.
    We load it once at startup and reuse it for every request.
    """

    def __init__(self):
        print(f"[Embedder] Loading model: {settings.EMBEDDING_MODEL}")
        self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        print("[Embedder] Model loaded successfully.")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of text strings into a list of float vectors.
        Returns: list of embeddings, one per input text.
        """
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()


# Singleton: loaded once when the module is first imported
embedder = Embedder()
