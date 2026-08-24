from tavily import TavilyClient
from config import TAVILY_KEY

client = TavilyClient(TAVILY_KEY)


def web_search(
    query: str,
    search_depth: str = "advanced",
    max_results: int = 5,
    topic: str = "general"
) -> str:

    response = client.search(
        query=query,
        search_depth=search_depth,
        max_results=max_results,
        topic=topic
    )

    context = "\n\n".join(
        result["content"]
        for result in response["results"]
    )

    return context


if __name__ == "__main__":

    query = "what happened about today's Karnataka bandh"

    context = web_search(query)

    print(context)