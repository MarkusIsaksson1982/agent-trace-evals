"""A small seeded agent useful for exercising a trace evaluation harness."""
from __future__ import annotations

import ast
import operator
import random
import re
import time
from typing import Any, Callable

from .trace import Span, Trace

_WORDS = re.compile(r"[a-z0-9]+")
_CITATION = re.compile(r"\[doc:([\w-]+)\]")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _calculate(expression: str) -> float | int:
    tree = ast.parse(expression, mode="eval")
    def visit(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression): return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise ValueError("unsupported arithmetic")
    return visit(tree)


def _toy_llm(question: str, documents: list[tuple[str, str]], calculation: Any = None) -> str:
    if calculation is not None:
        return f"The answer is {calculation}."
    if not documents:
        return "I don't know."
    doc_id, text = documents[0]
    lower = question.lower()
    if "refund" in lower:
        text = "Refunds are available within 30 days of purchase when the item is unused."
    elif "warranty" in lower:
        text = "Products include a one-year limited warranty covering manufacturing defects."
    return f"{text} [doc:{doc_id}]"


def run_agent(question: str, kb: dict[str, str], seed: int, flakiness: float = 0.0,
              llm: Callable[..., str] | None = None, case_id: str = "", run_idx: int = 0) -> Trace:
    """Run keyword retrieval, optional arithmetic, and answer generation."""
    if not 0 <= flakiness <= 1:
        raise ValueError("flakiness must be between 0 and 1")
    rng = random.Random(seed)
    spans: list[Span] = []
    start = time.perf_counter() * 1000
    question_words = set(_WORDS.findall(question.lower()))
    is_math = bool(re.search(r"\d\s*[+*/-]\s*\d", question))
    is_out_of_kb = any(marker in question.lower() for marker in ("population of mars", "capital of atlantis", "invented fact"))
    retrieved: list[tuple[str, str]] = []
    if not is_math:
        t0 = time.perf_counter() * 1000
        ranked = sorted(((len(question_words & set(_WORDS.findall(text.lower()))), key, text) for key, text in kb.items()), reverse=True)
        if ranked and ranked[0][0] > 0 and not is_out_of_kb:
            _, doc_id, text = max(ranked, key=lambda row: (row[0], row[1]))
            retrieved = [(doc_id, text)]
        if flakiness and rng.random() < flakiness * 0.55:
            retrieved = []
        spans.append(Span("retriever", "retrieval", question, [key for key, _ in retrieved], t0,
                          time.perf_counter() * 1000))

    calculation = None
    if is_math:
        match = re.search(r"(-?\d+(?:\.\d+)?)\s*([+*/-])\s*(-?\d+(?:\.\d+)?)", question)
        expression = "".join(match.groups()) if match else ""
        tool_name = "calculator"
        wrong_tool = bool(flakiness and rng.random() < flakiness * 0.55)
        if wrong_tool: tool_name = "search"
        t0 = time.perf_counter() * 1000
        try:
            if tool_name == "calculator": calculation = _calculate(expression)
            else: raise ValueError("wrong tool selected")
            tool_output = calculation
            error = None
        except Exception as exc:
            tool_output, error = None, str(exc)
        spans.append(Span(tool_name, "tool", expression, tool_output, t0,
                          time.perf_counter() * 1000, error=error))

    generator = llm or _toy_llm
    t0 = time.perf_counter() * 1000
    try:
        answer = str(generator(question, retrieved, calculation))
        error = None
    except Exception as exc:
        answer, error = "", str(exc)
    if flakiness and rng.random() < flakiness * 0.5:
        answer += " [doc:unknown]"
    spans.append(Span("answer", "llm", question, answer, t0, time.perf_counter() * 1000,
                      tokens_in=len(question.split()), tokens_out=len(answer.split()),
                      cost_usd=(len(question.split()) + len(answer.split())) * 0.000001, error=error))
    return Trace(case_id, run_idx, seed, spans, answer)
