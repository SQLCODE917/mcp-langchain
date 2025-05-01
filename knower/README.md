# KNOWER

Converse with your codebase

## Requirements

- Read and index a local codebase
- Support semantic search (via embeddings or keywords)
- Answer questions by retrieving relevant context and feeding it to the LLM

Embeddings and Semantic Search

- Generate embeddings for each code chunk using `langchain.embeddings`
- Store in vector DB
- During tool execution, do a similarity search against the user query

## Serious Python developers only

The complexity grows, time for some tooling:

```bash
pip install pylint
```

so you can do `pylint server.py`

## Adding Semantic Code Search

New dependencies since the `forerunner` and `prototype`

```bash
pip install --upgrade langchain langchain-community
pip install faiss-cpu tiktoken
pip install -U langchain-ollama
```

In Ollama, install [nomic-embed-text](https://ollama.com/library/nomic-embed-text)

### Explicit pickle loading

`FAISS.load_local` errors out with `ValueError: The de-serialization relies on loading a pickle file...`

Fix in `server.py`:

```python
VECTORSTORE = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
```

## Axiomatic Understandings

- [MCP Prompting](https://modelcontextprotocol.io/docs/concepts/prompts)
- [ReAct Prompting](https://www.promptingguide.ai/techniques/react)
- [LangChain API Reference](https://python.langchain.com/api_reference/core/index.html)
- [Reference Codebase](https://github.com/arunpshankar/react-from-scratch)

## Design Decisions

- Using `nvidia_Llama-3.1-8B-UltraLong-4M-Instruct-Q6_K_L` to explore the limits of context size
- Modelfile to
    - prevent infinite looping
    - integrate Ollama with LangChain's StructuredTool format
- in `server.py`, when building the FAISS index, experimenting with chunk size and quantity
- in `server.py`, `recursion_limit` to avoid infinite looping
- in `client.py`, the `explainer_prompt` which prioritizes tool use over defaulting to internal knowledge

## Optimizing chunking for Python

Instead of splittinig every x characters, split along function and class definitions using an AST

Scaling issue: chunking by top-level AST nodes is simple, but not optimal for large or deeply nested files.
Maybe need to implement hybrid chunking (AST + semantic text splitting)
and store the context hierarchy (class -> method -> docstring) in metadata?

2025-04-30

This did have limitations:
- Naive recursive character-based splitting of chunks may fragment cohesive logic
- AST Incompleteness, like ignoring module-level logic, lambdas, or nested expressions not wrapped in class/function definitions

Mitigated by:
- Creating a new AST-informed chunking function that merges docstrings and nested blocks
- Which introduced a new hardware and LLM limit - large classes and functions had no size constraint, so
- Splitting AST chunks if they are oversized, but still preserving semantic boundaries
- Including non-AST top-level logic in the chunks
- Including inline comments

## Known limitations

- Does not support polyglot codebases - .py only
- Only supports similarity-based search with no symbolic resolution, no callgraph traversal, and no dependency awareness
- No hot-reloading on codebase changes
