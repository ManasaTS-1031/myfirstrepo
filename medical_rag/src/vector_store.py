import chromadb
from config import CHROMA_PATH, COLLECTION_NAME, DISTANCE_METRIC


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": DISTANCE_METRIC}
    )


def upsert_vectors(ids, documents, embeddings, metadatas):
    collection = get_collection()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Upserted {len(ids)} vectors into '{COLLECTION_NAME}'")


if __name__ == "__main__":

    print("Starting ingestion...")

    from ingest_data import readdoc, chunk_text

    filepath = r"C:\rag\medical_rag\data\rag_common cold.pdf"

    print("PDF path:", filepath)

    print("Reading PDF...")

    text = readdoc(filepath)

    print("PDF read successfully.")
    print("Creating chunks and embeddings...")

    ids, documents, embeddings, metadatas = chunk_text(
        text,
        filepath
    )

    print(f"Created {len(documents)} chunks.")

    print("Saving vectors to ChromaDB...")

    upsert_vectors(
        ids,
        documents,
        embeddings,
        metadatas
    )

    print("Ingestion completed!")