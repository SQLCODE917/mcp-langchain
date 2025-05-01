import ast
import hashlib
from pathlib import Path
from typing import List
from langchain.docstore.document import Document

def attach_parents(tree: ast.AST):
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

def find_parent_class(node: ast.AST):
    while hasattr(node, 'parent'):
        node = node.parent
        if isinstance(node, ast.ClassDef):
            return node.name
    return None


def extract_ast_chunk(code: str, lines: List[str], node: ast.AST) -> str:
    """
    Returns full chunk for a node, including decorators and docstring.
    """
    start = min([getattr(node, 'lineno', 1)] + [getattr(d, 'lineno', 1) for d in getattr(node, 'decorator_list', [])]) - 1
    end = getattr(node, 'end_lineno', start + 1)
    return "".join(lines[start:end])

def extract_chunks(filepath: Path) -> List[Document]:
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    try:
        tree = ast.parse(code)
        attach_parents(tree)
    except SyntaxError:
        print(f"Skipping file with syntax error: {filepath}")
        return []

    lines = code.splitlines(keepends=True)
    documents = []
    visited_spans = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            chunk = extract_ast_chunk(code, lines, node)
            start = min([getattr(node, 'lineno', 1)] + [getattr(d, 'lineno', 1) for d in getattr(node, 'decorator_list', [])]) - 1
            end = getattr(node, 'end_lineno', start + 1)

            visited_spans.add((start, end))
            chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()[:8]

            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": str(filepath),
                    "symbol_type": node.__class__.__name__.lower(),
                    "symbol_name": getattr(node, 'name', 'unknown'),
                    "chunk_id": f"{filepath}::{node.__class__.__name__.lower()}-{getattr(node, 'name', 'unknown')}-{chunk_hash}",
                    "start_line": start,
                    "end_line": end,
                    "parent_class": find_parent_class(node)
                }
            ))

    # Include top-level logic not captured in AST functions/classes
    covered_lines = set(i for start, end in visited_spans for i in range(start, end))
    top_level_lines = [line for i, line in enumerate(lines) if i not in covered_lines]

    if top_level_lines:
        top_chunk = "".join(top_level_lines).strip()
        if top_chunk:
            chunk_hash = hashlib.md5(top_chunk.encode("utf-8")).hexdigest()[:8]
            documents.append(Document(
                page_content=top_chunk,
                metadata={
                    "source": str(filepath),
                    "symbol_type": "toplevel",
                    "symbol_name": "__toplevel__",
                    "chunk_id": f"{filepath}::toplevel-{chunk_hash}",
                    "start_line": 0,
                    "end_line": len(lines),
                    "parent_class": None
                }
            ))

    return documents

