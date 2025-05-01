"""MCP server that provides a codebase semantic search tool via FAISS and LangChain."""

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
import ast
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from server.recursive_character_text_splitter import extract_chunks

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

@mcp.tool()
def search_codebase(query: str) -> str:
    """
    Search the codebase for relevant code using semantic embeddings.

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
    matches = VECTORSTORE.similarity_search(query, k=10)
    if not matches:
        return "No relevant code found."

    logger.info("Returning %d documents from FAISS search.", len(matches))

    # debug chunk logging
    for match_index, d in enumerate(matches, 1):
        logger.info("Match %d: %s", match_index, d.metadata.get("chunk_id", "unknown"))

    return "\n\n".join([
        f"File: {d.metadata.get('source', 'unknown')}\n---\n{d.page_content[:800]}..."
        for d in matches if d.page_content
    ])

def main():
    """Start the Knower MCP Server."""
    logger.info("Starting the Knower MCP Server...")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
