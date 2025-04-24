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

## Resign Decisions

- Using `nvidia_Llama-3.1-8B-UltraLong-4M-Instruct-Q6_K_L` to explore the limits of context size
- Modelfile with a template to enable tool use and prevent infinite looping
- in `server.py`, when building the FAISS index, experimenting with chunk size and quantity
- in `server.py`, `recursion_limit` to avoid infinite looping
- in `client.py`, the `explainer_prompt` which prioritizes tool use over defaulting to internal knowledge
