import os
import pickle

import torch

from embed import embedfunc, sparse_embedfunc
from vector_store import get_collection


SPARSE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sparse_embeddings.pkl")


def _embedding_to_dict(embedding):
    if not isinstance(embedding, torch.Tensor):
        embedding = torch.as_tensor(embedding)
    if embedding.layout == torch.sparse_coo:
        embedding = embedding.coalesce()
        return dict(zip(embedding.indices()[-1].tolist(), embedding.values().tolist()))
    values = embedding.reshape(-1)
    indices = torch.nonzero(values, as_tuple=False).reshape(-1)
    return dict(zip(indices.tolist(), values[indices].tolist()))


def sparse_similarity(query_sparse, document_sparse):
    query_values = _embedding_to_dict(query_sparse)
    document_values = _embedding_to_dict(document_sparse)
    return sum(
        value * document_values.get(index, 0.0)
        for index, value in query_values.items()
    )


def sparse_retrieve(query, top_k=20, docname_filter=None):
    with open(SPARSE_PATH, "rb") as file:
        sparse_store = pickle.load(file)
    query_sparse = sparse_embedfunc(query)
    results = []
    for doc_id, embedding in sparse_store.items():
        if docname_filter and not doc_id.startswith(f"{docname_filter}_"):
            continue
        results.append((doc_id, sparse_similarity(query_sparse, embedding)))
    return sorted(results, key=lambda item: item[1], reverse=True)[:top_k]


def dense_retrieve(query, top_k=20, docname_filter=None):
    where = {"docname": docname_filter} if docname_filter else None
    results = get_collection().query(
        query_embeddings=[embedfunc(query)],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )
    return [
        {"id": doc_id, "text": text, "docname": metadata["docname"],
         "distance": distance, "similarity": 1 - distance}
        for doc_id, text, metadata, distance in zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0]
        )
    ]


def rrf_fusion(dense_results, sparse_results, top_k=5, rrf_k=60):
    fused = {}
    for rank, result in enumerate(dense_results, start=1):
        item = fused.setdefault(result["id"], {"score": 0.0, "dense": None, "sparse": None})
        item["score"] += 1 / (rrf_k + rank)
        item["dense"] = result
    for rank, (doc_id, score) in enumerate(sparse_results, start=1):
        item = fused.setdefault(doc_id, {"score": 0.0, "dense": None, "sparse": None})
        item["score"] += 1 / (rrf_k + rank)
        item["sparse"] = score
    output = []
    for doc_id, item in sorted(fused.items(), key=lambda pair: pair[1]["score"], reverse=True)[:top_k]:
        dense = item["dense"]
        output.append({
            "id": doc_id,
            "text": dense["text"] if dense else None,
            "docname": dense["docname"] if dense else None,
            "rrf_score": item["score"],
            "dense_similarity": dense["similarity"] if dense else None,
            "sparse_score": item["sparse"]
        })
    return output


def retrieve(query, top_k=5, candidate_k=20, docname_filter=None):
    dense = dense_retrieve(query, candidate_k, docname_filter)
    sparse = sparse_retrieve(query, candidate_k, docname_filter)
    fused = rrf_fusion(dense, sparse, top_k)

    missing_ids = [item["id"] for item in fused if item["text"] is None]
    if missing_ids:
        records = get_collection().get(
            ids=missing_ids,
            include=["documents", "metadatas"]
        )
        records_by_id = {
            record_id: (document, metadata)
            for record_id, document, metadata in zip(
                records["ids"], records["documents"], records["metadatas"]
            )
        }
        for item in fused:
            record = records_by_id.get(item["id"])
            if record:
                item["text"] = record[0]
                item["docname"] = record[1]["docname"]

    return fused


if __name__ == "__main__":
    for match in retrieve("what are the symptoms of cold"):
        print(f"[RRF={match['rrf_score']:.4f}] ({match['docname']})")
        print(match["text"])