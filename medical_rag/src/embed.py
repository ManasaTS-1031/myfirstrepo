from sentence_transformers import SentenceTransformer, SparseEncoder
from config import embedding_model


_model = None
_sparsemodel = None


def get_model():
    global _model

    if _model is None:
        print("Loading dense embedding model...")
        _model = SentenceTransformer(embedding_model)

    return _model


def get_sparsemodel():
    global _sparsemodel

    if _sparsemodel is None:
        print("Loading sparse embedding model...")
        _sparsemodel = SparseEncoder(
            "naver/splade-cocondenser-ensembledistil"
        )

    return _sparsemodel


def embedfunc(sent):
    model = get_model()
    return model.encode(sent).tolist()


def sparse_embedfunc(sent):
    model = get_sparsemodel()

    sparse = model.encode(sent)

    # A single string may already produce a one-dimensional embedding.
    return sparse[0] if sparse.ndim > 1 else sparse