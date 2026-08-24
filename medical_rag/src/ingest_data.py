import pymupdf
import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embed import embedfunc
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)



def readdoc(filepath):
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text


def chunk_text(text, filepath):
    chunks = text_splitter.split_text(text)
    docname = os.path.basename(filepath)

    ids, documents, embeddings, metadatas = [], [], [], []

    for i, chunk in enumerate(chunks):
        embedding = embedfunc(chunk)
        ids.append(f"{docname}_{i}")
        documents.append(chunk)
        embeddings.append(embedding)
        metadatas.append({"docname": docname})

    return ids, documents, embeddings, metadatas