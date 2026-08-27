import os

from dotenv import load_dotenv
from retrieval import retrieve
from llm_response import llm_chat
from tavily import TavilyClient
from config import THRESHOLD

load_dotenv()

TAVILY_KEY = os.getenv("TAVILY_KEY")

if not TAVILY_KEY:
    raise ValueError(
        "TAVILY_KEY is missing from your .env file"
    )

tavily_client = TavilyClient(
    api_key=TAVILY_KEY
)

def search_web(query: str, max_results: int = 3):

    response = tavily_client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced"
    )

    results = response.get("results", [])

    context_parts = []
    citations = []

    for i, result in enumerate(results, start=1):

        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content", "")

        context_parts.append(
            f"""
WEB SOURCE {i}
Title: {title}
URL: {url}
Content:
{content}
"""
        )

        citations.append({
            "type": "web",
            "title": title,
            "url": url
        })

    context = "\n\n".join(context_parts)

    return context, citations


def adaptive_rag_pipeline(
    user_question: str,
    threshold: THRESHOLD
):

    # --------------------------------------------------
    # 1. SEARCH CHROMADB
    # --------------------------------------------------

    matches = retrieve(
        user_question,
        top_k=5
    )

    citations = []

    if not matches:

        print("No local results.")

        local_context = ""

        best_similarity = 0.0

    else:

        best_similarity = matches[0]["dense_similarity"] or 0.0

        print(
            f"Best similarity: {best_similarity:.4f}"
        )

        local_context = "\n\n".join(
            match["text"]
            for match in matches
        )

        # Local citations
        # --------------------------------------------------
# LOCAL CITATIONS
# --------------------------------------------------

    seen_local_docs = set()

    for match in matches:

        docname = match["docname"]

        if docname not in seen_local_docs:

            citations.append({
                "type": "local",
                "title": docname,
                "similarity": match["dense_similarity"] or 0.0
            })

            seen_local_docs.add(docname) 
    # --------------------------------------------------
    # 2. ADAPTIVE ROUTING
    # --------------------------------------------------

    if best_similarity >= threshold:

        print("Using local knowledge.")

        context = local_context

        source_type = "LOCAL"

    else:

        print("Low confidence.")
        print("Searching Tavily...")

        web_context, web_citations = search_web(
            user_question,
            max_results=3
        )

        # If local results exist, combine them
        if local_context:

            context = f"""
LOCAL MEDICAL KNOWLEDGE:

{local_context}


WEB KNOWLEDGE:

{web_context}
"""

        else:

            context = web_context

        citations.extend(web_citations)

        source_type = "LOCAL + WEB"


    # --------------------------------------------------
    # 3. GENERATE GEMINI ANSWER
    # --------------------------------------------------

    answer = llm_chat(
        context=context,
        user_query=user_question
    )


    # --------------------------------------------------
    # 4. RETURN ANSWER + CITATIONS
    # --------------------------------------------------

    return {
        "answer": answer,
        "citations": citations,
        "source_type": source_type,
        "similarity": best_similarity
    }