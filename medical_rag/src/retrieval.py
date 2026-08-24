# src/retriever.py
from embed import embedfunc
from vector_store import get_collection

def retrieve(query, top_k=5, docname_filter=None):
    collection = get_collection()
    query_embedding = embedfunc(query)

    where = {"docname": docname_filter} if docname_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )   



    # print(text_rag)
    output = []
    outtext=[]
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        # with hnsw:space="cosine", Chroma returns cosine DISTANCE
        # similarity = 1 - distance
        similarity = 1 - dist
        output.append({
            "text": doc,
            "docname": meta["docname"],
            "distance": dist,
            "similarity": similarity
        })

    return output

# def retrieve(query, top_k=5, docname_filter=None):
#     collection = get_collection()

#     query_embedding = embedfunc(query)

#     where = {"docname": docname_filter} if docname_filter else None

#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k,
#         where=where,
#         include=["documents", "metadatas", "distances"]
#     )


#     text_rag = []

#     for doc in results["documents"][0]:
#         text_rag.append({
#             "text": doc
#         })

#     text = "\n\n".join(item["text"] for item in text_rag)
     
#     print("\n===== RETRIEVED CONTEXT =====")
#     print(text)
#     print("=============================\n")


#     return text
if __name__ == "__main__":
    query = """ what are the symptoms of cold"""
    matches = retrieve(query, top_k=5)


    for m in matches:
        print(f"[sim={m['similarity']:.4f}] ({m['docname']}) {m['text']}\n")