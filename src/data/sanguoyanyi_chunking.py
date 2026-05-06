"""《三国演义》原文：按「第 N 回」标题切分为块，供导出 jsonl / 后续 RAG 索引。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# 行首「第…回」直至行末为回目标题（含副题）
_CHAPTER_HEAD = re.compile(r"^第([零一二三四五六七八九十百千两]+)回\s*(.*)$")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def default_raw_txt_path() -> Path:
    return project_root() / "data" / "raw" / "三国演义（原文版）.txt"


def default_jsonl_path() -> Path:
    return project_root() / "data" / "kb" / "sanguoyanyi_chunks.jsonl"


def split_into_chapters(text: str) -> list[tuple[str, str]]:
    """按回目切分：返回 [(chapter_title_line, body_text), ...]。"""
    lines = text.splitlines()
    chapters: list[tuple[str, str]] = []
    title: str | None = None
    body: list[str] = []
    preamble: list[str] = []

    for line in lines:
        if _CHAPTER_HEAD.match(line):
            if title is not None:
                chapters.append((title, "\n".join(body).strip()))
            title = line.strip()
            body = []
            continue
        if title is None:
            preamble.append(line)
        else:
            body.append(line)

    if title is not None:
        chapters.append((title, "\n".join(body).strip()))

    pre = "\n".join(preamble).strip()
    if pre and chapters:
        t0, b0 = chapters[0]
        chapters[0] = (t0, f"{pre}\n\n{b0}".strip() if b0 else pre)
    return chapters


def chapters_to_chunk_dicts(
    chapters: list[tuple[str, str]],
    *,
    doc_id: str = "sanguoyanyi_v1",
) -> list[dict[str, Any]]:
    """转为可写入 jsonl 的记录（每回一条 chunk）。"""
    out: list[dict[str, Any]] = []
    for i, (chapter_title, body) in enumerate(chapters, start=1):
        chunk_id = f"{doc_id}_hui_{i:04d}"
        out.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chapter_seq": i,
                "chapter_title": chapter_title,
                "text": body,
            }
        )
    return out


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_chunks_from_file(path: Path, *, doc_id: str = "sanguoyanyi_v1") -> list[dict[str, Any]]:
    text = load_text(path)
    chapters = split_into_chapters(text)
    return chapters_to_chunk_dicts(chapters, doc_id=doc_id)
