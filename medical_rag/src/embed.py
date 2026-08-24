# src/embed.py
from sentence_transformers import SentenceTransformer
from config import embedding_model

_model = None  # module-level cache, loaded once

def get_model():
    global _model
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(embedding_model)
    return _model

def embedfunc(sent):
    model = get_model()
    embed = model.encode(sent)
    return embed.tolist()  # convert numpy array -> list, Chroma expects list-like