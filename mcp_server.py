"""
MCP server: lets Claude search the Coffee Board documents.

Claude writes the answer itself; this only finds the passages, using the same
hybrid search and reranker as the web app.

    python mcp_server.py            stdio, for Claude Code / Claude Desktop (.mcp.json)
    python mcp_server.py --http     HTTP at http://127.0.0.1:8001/mcp, for a tunnel
                                    or a public URL; add --allow-host for that URL
"""

import argparse

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

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
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--http", action="store_true",
                        help="serve over HTTP instead of stdio")
    # 8000 is taken by the web app (api.py).
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--allow-host", action="append", default=[],
                        metavar="HOST",
                        help="public hostname requests may arrive under, e.g. "
                             "abc123.ngrok-free.app (repeatable)")
    args = parser.parse_args()

    if not args.http:
        mcp.run()      # stdio: Claude Code starts it and talks to it directly
    else:
        # The Host header is checked to block DNS-rebinding attacks. Requests
        # through a tunnel carry the tunnel's hostname, so it must be allowed.
        hosts = ["127.0.0.1:*", "localhost:*", *args.allow_host]
        origins = [f"https://{h}" for h in args.allow_host]
        mcp.run(
            transport="streamable-http", host=args.host, port=args.port,
            transport_security=TransportSecuritySettings(
                allowed_hosts=hosts, allowed_origins=origins,
            ),
        )
