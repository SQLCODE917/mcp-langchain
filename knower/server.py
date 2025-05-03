"""MCP server that provides a codebase semantic search tool via FAISS and LangChain."""

import logging
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.docstore.document import Document
from sentence_transformers import CrossEncoder

from server.chunk_splitter import extract_chunks

# Logging to not clog MCP comms
os.makedirs("logs", exist_ok=True)
LOG_FILENAME = f"logs/server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILENAME, encoding="utf-8")
    ]
)
logger = logging.getLogger("knower_mcp_server")

mcp = FastMCP("CodebaseKnower")

# Global store so we don't re-index every time
VECTORSTORE = None

INDEX_DIR = "faiss_index"
EMBED_MODEL = "nomic-embed-text"
DEFAULT_REPO_PATH = Path(__file__).parent.parent / "prototype"
# this is the HuggingFace reranker model - must be good, right?
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER = CrossEncoder(RERANK_MODEL_NAME)

load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


# Load or build index once when server starts
if os.path.exists(INDEX_DIR):
    logger.info("Loading existing FAISS index on startup...")
    VECTORSTORE = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    logger.info("Building FAISS index on startup...")
    chunks = []
    for root, _, files in os.walk(DEFAULT_REPO_PATH):
        logger.info("Walking %s", root)
        for file in files:
            logger.info("Processing %s", file)
            if file.endswith(".py"):
                logger.info("Adding %s to docs", file)
                filepath = Path(root) / file
                file_chunks = extract_chunks(filepath)
                for doc in file_chunks:
                    logger.info("Indexing chunk %s", doc.metadata.get("chunk_id"))
                chunks.extend(file_chunks)

    VECTORSTORE = FAISS.from_documents(chunks, embeddings)
    VECTORSTORE.save_local(INDEX_DIR)
    logger.info("Index saved to %s", INDEX_DIR)

def rerank(query: str, docs: list[Document], top_k: int = 10) -> list[Document]:
    """
    Rerank a list of LangChain documents using a cross-encoder.

    Args:
        query: The user's search query.
        docs: A list of LangChain Document objects.
        top_k: Number of top reranked results to return.

    Returns:
        List of Documents sorted by relevance.
    """
    logger.info("Reranking %d documents", len(docs))
    pairs = [(query, doc.page_content) for doc in docs]
    scores = RERANKER.predict(pairs)

    # Combine scores with documents
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked[:top_k]]

@mcp.tool()
def search_codebase(query: str) -> str:
    """
    Search the codebase for relevant code using semantic embeddings with metadata filtering.

    Use this tool to locate:
    - Function and class definitions
    - Specifig logic or algorithms
    - Implementation details
    - Files that reference a keyword and concept

    Args:
        query: A natural language question about the codebase.

    Returns:
        Relevant code snippets or explanations based on the query.
    """
    logger.info("Invoking search_codebase tool for query %s", query)
    matches = VECTORSTORE.similarity_search(query, k=20)
    if not matches:
        return "No relevant code found."

    logger.info("1. Semantic search returned %d documents.", len(matches))
    # Debug chunk logging
    for match_index, d in enumerate(matches, 1):
        logger.info("Match %d: %s", match_index, d.metadata.get("chunk_id", "unknown"))

    # Infer filter intent from query experiment
    filter_types = []
    if "function" in query.lower():
        filter_types.append("function")
    if "class" in query.lower():
        filter_types.append("class")
    if "comment" in query.lower():
        filter_types.append("comment")
    if "top-level" in query.lower() or "script" in query.lower():
        filter_types.append("toplevel")

    if filter_types:
        logger.info("Applying metadata filter for types: %s", filter_types)
        filtered = [
            d for d in matches
            if d.metadata.get("symbol_type", "").lower() in filter_types
        ]
    else:
        filtered = matches

    if not filtered:
        return f"No relevant code found for types: {filter_types}"

    logger.info("2. Returning %d filtered documents.", len(filtered))

    # Reranking
    # Each query/chunks pair is scored by a cross-encoder that jointly attends to both inputs.
    # Warning: This is a blocking call.
    # Could be made async, if needed, I suppose,
    #   by wrapping with FastAPI+uvicorn to make an "inference server" or something...
    top_k = rerank(query, filtered, top_k=10)

    logger.info("3. Reranking:")
    for rerank_index, d in enumerate(top_k, 1):
        logger.info("Rank %d: :%s", rerank_index, d.metadata.get("chunk_id", "unknown"))

    return "\n\n".join([
        f"File: {d.metadata.get('source', 'unknown')}\nSymbol: {d.metadata.get('symbol_qualifier')}\n---\n{d.page_content[:800]}..."
        for d in top_k if d.page_content
    ])

def main():
    """Start the Knower MCP Server."""
    logger.info("Starting the Knower MCP Server...")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
