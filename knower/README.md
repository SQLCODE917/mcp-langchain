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

2025-05-03

Considering the semantic chunking system strong enough, attempting to now to introduce ranking (like, reranking and filtering),
and narrowing (like, by looking at metadata and having a chunk selection strategy):

1. Upgrade from Similarity Search to Hybrid Reranking:
adding a reranker to sort top-k results using query context matching.
This should make good use of the fine-grained chunks and their metadata!
I expect signifficant improvements.
```
pip install sentence-transformers
```
Error: `ERROR: Error while checking for conflicts. ... TypeError: 'str' object is not callable`
Workaround (maybe downgrade pip from 25.0 to a version from Python 3.10 era?)
```
pip install "pip<23.3"
pip install sentence-transformers
```

2. Filter using `symbol_type`, symbol_name` and `parent_class`:
going to invent criteria like "if query includes `class`, boost when `symbol_type == 'classdef'`,
and if query includes "in function X", filter by `symbol_name == X`.
Hoping for increased precision.

Success! chunk debugging showed that for exact match, number of chunks was reduced from 11 to 4(!),
and for open ended queries, the ranking was improved by boosting more relevant chunks.

Addendum:
Introduced a chunking lower limit to keep small functions as is.
512 tokens is (assuming 1.2~1.5 tokens per word), 350~420 words, maybe 30~40 lines of Python code.
That's pretty much the ideal function size, isn't it?
Keep those intact and don't split them into multiple chunks.


## Known limitations

- Does not support polyglot codebases - .py only
- Only supports similarity-based search with no symbolic resolution, no callgraph traversal, and no dependency awareness
- No hot-reloading on codebase changes
