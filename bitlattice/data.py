"""Training data sources yielding (input_ids, labels) batches of a fixed shape.

Every source is deterministic given its seed and reads its input in a fixed order, so all processes of a pipeline
run build identical batches without exchanging data.

    TextWindows(ids, seq, batch)          random windows of a token stream (plain-text corpora)
    JsonlStream(path, tokenizer, seq, batch)
                                          prompt / response records from a JSONL file that may still be growing
                                          (written by a producer process); the loss covers the responses only,
                                          examples are packed into rows of ``seq`` + 1 tokens
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator

import torch

__all__ = ["IGNORE", "JsonlStream", "TextWindows", "encode_example"]

IGNORE = -100


class TextWindows:
    """Random windows of ``seq`` + 1 tokens from a 1-D token tensor; labels are the inputs."""

    def __init__(self, ids: torch.Tensor, seq: int, batch: int, seed: int = 0):
        self.ids, self.seq, self.batch = ids, seq, batch
        self.g = torch.Generator().manual_seed(1000 + seed)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            s = torch.randint(0, self.ids.numel() - self.seq - 1, (self.batch,), generator=self.g)
            x = torch.stack([self.ids[i:i + self.seq + 1] for i in s])
            yield x, x


def encode_example(tokenizer, record: dict) -> tuple[list[int], list[int]]:
    """(tokens, labels) of one record. ``{"prompt", "response"}`` or ``{"messages": [{"role", "content"}, ...]}``
    (the last assistant message is the target). The chat template of the tokenizer is used when it has one."""
    if "messages" in record:
        msgs = record["messages"]
        prompt_msgs, target = msgs[:-1], msgs[-1]["content"]
    else:
        prompt_msgs, target = [{"role": "user", "content": record["prompt"]}], record["response"]
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join(m["content"] for m in prompt_msgs) + "\n"
    p = tokenizer(prompt, add_special_tokens=False).input_ids
    eos = tokenizer.eos_token or ""
    t = tokenizer(target + eos, add_special_tokens=False).input_ids
    return p + t, [IGNORE] * len(p) + t


class JsonlStream:
    """Packed prompt/response batches from a JSONL file, read in order; waits for new lines while the file grows
    (``follow``), or cycles over it. Examples longer than a row are truncated from the left of the prompt."""

    def __init__(self, path: str, tokenizer, seq: int, batch: int, follow: bool = True, poll_s: float = 1.0,
                 pad_id: int | None = None):
        self.path, self.tok, self.seq, self.batch = path, tokenizer, seq, batch
        self.follow, self.poll_s = follow, poll_s
        self.pad = pad_id if pad_id is not None else (tokenizer.pad_token_id or tokenizer.eos_token_id or 0)
        self.consumed = 0

    def _records(self) -> Iterator[dict]:
        while True:
            with open(self.path, encoding="utf-8") as f:
                n = 0
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line.endswith("\n"):                     # end of file or a half-written line
                        if not self.follow:
                            break
                        f.seek(pos)
                        time.sleep(self.poll_s)
                        continue
                    n += 1
                    if line.strip():
                        self.consumed += 1
                        yield json.loads(line)
            if n == 0:                                              # not following: cycle, unless empty
                raise RuntimeError(f"{self.path} has no records")

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        L = self.seq + 1
        rows_x, rows_y = [], []
        cur_x, cur_y = [], []
        for rec in self._records():
            x, y = encode_example(self.tok, rec)
            if len(x) > L:                                          # keep the target, cut the prompt's start
                x, y = x[-L:], y[-L:]
            if len(cur_x) + len(x) > L:
                pad = L - len(cur_x)
                rows_x.append(cur_x + [self.pad] * pad)
                rows_y.append(cur_y + [IGNORE] * pad)
                cur_x, cur_y = [], []
                if len(rows_x) == self.batch:
                    yield torch.tensor(rows_x), torch.tensor(rows_y)
                    rows_x, rows_y = [], []
            cur_x += x
            cur_y += y


def wait_for(path: str, timeout_s: float = 600.0) -> None:
    """Block until ``path`` exists (a producer may still be starting)."""
    t = time.time()
    while not os.path.exists(path):
        if time.time() - t > timeout_s:
            raise TimeoutError(path)
        time.sleep(1.0)
