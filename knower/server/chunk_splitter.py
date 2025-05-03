"""Chunk splitter for Python source code using AST and token analysis."""

import ast
import hashlib
import tokenize
from pathlib import Path
from typing import List
from io import StringIO
from langchain.docstore.document import Document

def attach_parents(tree: ast.AST):
    """Annotate AST nodes with their parent nodes."""
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

def find_parent_class(node: ast.AST):
    """Find the nearest parent class name for a given AST node."""
    while hasattr(node, 'parent'):
        node = node.parent
        if isinstance(node, ast.ClassDef):
            return node.name
    return None

def extract_ast_chunk(lines: List[str], node: ast.AST) -> str:
    """Extract source lines corresponding to a class or function AST node."""
    start = min(
        [getattr(node, 'lineno', 1)] +
        [getattr(d, 'lineno', 1) for d in getattr(node, 'decorator_list', [])]
    ) - 1
    end = getattr(node, 'end_lineno', start + 1)
    return "".join(lines[start:end]), start, end

def extract_inline_comments(lines: List[str], covered_lines: set) -> List[str]:
    """Extract comments from uncovered lines."""
    comments = []
    try:
        code = "".join(lines)
        tokens = tokenize.generate_tokens(StringIO(code).readline)
        for toknum, tokval, (srow, _), (_, _), _ in tokens:
            if toknum == tokenize.COMMENT and (srow - 1) not in covered_lines:
                comments.append(tokval.strip())
    except tokenize.TokenError:
        pass
    return comments

def split_large_chunk(chunk: str, max_tokens: int = 512) -> List[str]:
    """Split a large chunk into smaller parts based on token count."""
    lines = chunk.splitlines(keepends=True)
    chunks = []
    buffer = []
    token_count = 0
    for line in lines:
        line_tokens = len(line.split())
        if token_count + line_tokens > max_tokens:
            chunks.append("".join(buffer).strip())
            buffer = [line]
            token_count = line_tokens
        else:
            buffer.append(line)
            token_count += line_tokens
    if buffer:
        chunks.append("".join(buffer).strip())
    return [c for c in chunks if c]

def process_toplevel_chunks(filepath: Path, lines: List[str], visited_spans: set) -> List[Document]:
    """Extract and return top-level code outside of functions/classes."""
    documents = []
    covered_lines = set(i for start, end in visited_spans for i in range(start, end))
    top_level_lines = [line for i, line in enumerate(lines) if i not in covered_lines]
    if not top_level_lines:
        return documents

    top_chunks = split_large_chunk("".join(top_level_lines).strip())
    for idx, sub in enumerate(top_chunks):
        chunk_hash = hashlib.md5(sub.encode("utf-8")).hexdigest()[:8]
        documents.append(Document(
            page_content=sub,
            metadata={
                "source": str(filepath),
                "symbol_type": "toplevel",
                "symbol_name": "__toplevel__",
                "chunk_id": f"{filepath}::toplevel-{chunk_hash}-{idx}",
                "start_line": 0,
                "end_line": len(lines),
                "parent_class": None
            }
        ))
    return documents

def process_inline_comments(filepath: Path, lines: List[str], visited_spans: set) -> List[Document]:
    """Extract inline comments from lines not already chunked."""
    documents = []
    covered_lines = set(i for start, end in visited_spans for i in range(start, end))
    inline_comments = extract_inline_comments(lines, covered_lines)
    for idx, comment in enumerate(inline_comments):
        chunk_hash = hashlib.md5(comment.encode("utf-8")).hexdigest()[:8]
        documents.append(Document(
            page_content=comment,
            metadata={
                "source": str(filepath),
                "symbol_type": "comment",
                "symbol_name": "__inline_comment__",
                "chunk_id": f"{filepath}::comment-{chunk_hash}-{idx}",
                "start_line": 0,
                "end_line": 0,
                "parent_class": None
            }
        ))
    return documents

def extract_chunks(filepath: Path) -> List[Document]:
    """Extract and return top-level code outside of functions/classes."""
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
            chunk, start, end = extract_ast_chunk(lines, node)
            visited_spans.add((start, end))

            parent_class = find_parent_class(node)
            # Add symbol_qualifier (e.g., ClassName.function_name)
            #   for deeper query filtering or reranking
            symbol_type = node.__class__.__name__.lower().replace("def", "")
            symbol_name = getattr(node, 'name', 'unknown')
            # Add symbol_qualifier (e.g., ClassName.function_name)
            #   for deeper query filtering or reranking
            symbol_qualifier = f"{parent_class}.{symbol_name}" if parent_class else symbol_name

            # Skip chunk splitting if the function/class is already under the token budget
            sub_chunks = (
                [chunk] if len(chunk.split()) < 512 else split_large_chunk(chunk)
            )
            for idx, sub in enumerate(sub_chunks):
                chunk_hash = hashlib.md5(sub.encode("utf-8")).hexdigest()[:8]
                documents.append(Document(
                    page_content=sub,
                    metadata={
                        "source": str(filepath),
                        # used to filter matches based on query heuristics
                        "symbol_type": symbol_type,
                        "symbol_name": symbol_name,
                        # used to filter matches based on query heuristics
                        "symbol_qualifier": symbol_qualifier,
                        "chunk_id": f"{filepath}::{symbol_type}-{symbol_name}-{chunk_hash}-{idx}",
                        "start_line": start,
                        "end_line": end,
                        "parent_class": parent_class
                    }
                ))

    documents += process_toplevel_chunks(filepath, lines, visited_spans)
    documents += process_inline_comments(filepath, lines, visited_spans)

    return documents
