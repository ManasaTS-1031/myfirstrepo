import pymupdf
import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embed import embedfunc, sparse_embedfunc
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

    ids = []
    documents = []
    dense_embeddings = []
    sparse_embeddings = []
    metadatas = []

    for i, chunk in enumerate(chunks):

        dense = embedfunc(chunk)
        sparse = sparse_embedfunc(chunk)

        ids.append(f"{docname}_{i}")
        documents.append(chunk)

        dense_embeddings.append(dense)
        sparse_embeddings.append(sparse)

        metadatas.append({
            "docname": docname
        })

    return (
        ids,
        documents,
        dense_embeddings,
        sparse_embeddings,
        metadatas
    )

if __name__=="__main__":
    r=readdoc(r"C:\rag\medical_rag\data\google ppr.pdf")
    a,b,c,d,e=chunk_text(r,filepath=r"C:\rag\medical_rag\data\google ppr.pdf")
    print(a)
    print(b)
    print(c)
    print(d)
    print(e)