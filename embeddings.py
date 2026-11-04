"""
Local embedding model (no API key needed).
 
all-MiniLM-L6-v2 turns text into 384-dim vectors. normalize_embeddings=True
makes cosine similarity work cleanly. The model downloads once (~90MB) on first use.
"""
 
from sentence_transformers import SentenceTransformer
 
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
 
 
def embed(texts):
    """Embed a single string or a list of strings. Returns numpy array(s)."""
    single = isinstance(texts, str)
    vecs = _model.encode([texts] if single else list(texts),
                         normalize_embeddings=True)
    return vecs[0] if single else vecs
 