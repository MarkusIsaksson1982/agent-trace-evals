"""Pluggable LLM clients. MockLLM is deterministic (tests); OpenCodeLLM shells out to a free OpenCode model."""
from __future__ import annotations
import re, subprocess, tempfile, time
from dataclasses import dataclass

@dataclass
class LLMResult:
    text: str; latency_s: float; prompt_tokens: int; completion_tokens: int; cost_usd: float = 0.0; model: str = ""

HEADER = re.compile(r"^> \S+ · \S+\s*$")

def est_tokens(s: str) -> int: return max(1, round(len(s) / 4))  # ~4 chars/token heuristic

class MockLLM:
    """Answers with the first sentence of the top context chunk + its citation; refuses if no context."""
    model = "mock"
    def complete(self, prompt: str) -> LLMResult:
        m = re.search(r"\[(\S+?)\] [^.\n]*\. (.+?[.!?])(?:\s|$)", prompt.split("CONTEXT:", 1)[-1])
        text = f"{m.group(2)} [{m.group(1)}]" if m else "I don't know based on the available documents."
        return LLMResult(text, 0.0, est_tokens(prompt), est_tokens(text), 0.0, self.model)

class OpenCodeLLM:
    def __init__(self, model="opencode/deepseek-v4.1-flash-free", timeout=180): self.model, self.timeout = model, timeout
    def complete(self, prompt: str) -> LLMResult:
        p = "Do not use any tools, do not read or write files. Reply with the answer text only.\n\n" + prompt
        t0 = time.time()
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run(["opencode", "run", "-m", self.model, p], cwd=d, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=self.timeout)
        if r.returncode != 0 or not r.stdout.strip():
            raise RuntimeError(f"opencode run failed (exit {r.returncode}) for {self.model}: {(r.stderr or '').strip()[-500:]}")
        text = re.sub(r"\x1b\[[0-9;]*m", "", r.stdout)
        # drop only the CLI's agent/model header ("> build · model"), not answer lines that happen to start with "> "
        text = "\n".join(l for l in text.splitlines() if not HEADER.match(l)).strip()
        return LLMResult(text, time.time() - t0, est_tokens(p), est_tokens(text), 0.0, self.model)
