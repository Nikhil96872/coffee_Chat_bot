"""
MCP server: lets Claude search the Coffee Board documents.

Claude writes the answer itself; this only finds the passages, using the same
hybrid search and reranker as the web app.
"""

from mcp.server.mcpserver import MCPServer

from retrieval import COLLECTION, Retriever

mcp = MCPServer("coffee-docs")
retriever = Retriever()


@mcp.tool()
def search(query: str, top_k: int = 5) -> list[dict]:
    """Search Indian Coffee Board documents (a grower's handbook and research
    abstracts) on coffee cultivation: varieties, planting, shade, nutrition,
    irrigation, pests, diseases, harvesting and processing.

    Answer only from the returned passages and cite each one by its citation.
    An empty list means the documents do not cover the question.
    """
    return [
        {
            "id": h.chunk_id,
            "citation": h.citation,
            "title": h.paper["title"] if h.paper else h.heading,
            "authors": h.paper["citation"] if h.paper else "",
            "text": h.passage,
            "score": round(h.rerank_score, 2),
        }
        for h in retriever.search(query, top_k=top_k)
    ]


@mcp.tool()
def fetch(id: int) -> dict:
    """Get the full text of one passage returned by search, by its id."""
    points = retriever.client.retrieve(COLLECTION, ids=[id], with_payload=True)
    if not points:
        return {"error": f"no passage with id {id}"}
    p = points[0].payload
    paper = retriever.papers.get(p.get("paper_id"))
    return {
        "id": id,
        "source": p["source"],
        "pages": f"{p['page_start']}-{p['page_end']}",
        "title": paper["title"] if paper else p["heading"],
        "text": paper["text"] if paper else p["text"],
    }


if __name__ == "__main__":
    mcp.run()          # stdio: Claude Code starts it and talks to it directly
