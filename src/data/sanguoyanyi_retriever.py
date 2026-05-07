"""《三国演义》jsonl 摘录：与三国地点检索同路径调用，词法匹配供回答引用。

不依赖外网 embedding；命中条数少时全量扫内存即可。
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.data.rag_retriever import _rag_debug, _tokens
from src.data.sanguoyanyi_chunking import default_jsonl_path


def _jsonl_path() -> Path:
    raw = os.getenv("SANGUOYANYI_JSONL", "").strip()
    if raw:
        return Path(raw).expanduser()
    return default_jsonl_path()


@lru_cache(maxsize=1)
def _load_all_chunks() -> tuple[list[dict[str, Any]], str]:
    """返回 (chunks, path_str)；文件缺失时 chunks=[]。"""
    path = _jsonl_path()
    if not path.is_file():
        _rag_debug(f"演义摘录：jsonl 不存在，跳过 path={path.resolve()}")
        return [], str(path.resolve())
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    _rag_debug(f"演义摘录：已加载 chunks={len(rows)} path={path.resolve()}")
    return rows, str(path.resolve())


def _score_chunk(text: str, q_tokens: list[str], place_names: list[str]) -> float:
    t = (text or "").lower()
    s = 0.0
    for pn in place_names:
        pn = (pn or "").strip().lower()
        if len(pn) >= 2 and pn in t:
            s += 5.0
    for tok in q_tokens:
        if len(tok) >= 2 and tok in t:
            s += 1.2
    return s


def search_romance_excerpts(
    query: str,
    place_names: list[str],
    *,
    top_k: int = 2,
    excerpt_max_chars: int = 480,
) -> list[dict[str, Any]]:
    """词法检索演义块：供三国地点查询同路径附加。返回带 excerpt 的字典列表。"""
    q = (query or "").strip()
    names = [str(x).strip() for x in place_names if str(x).strip()]
    if not q and not names:
        _rag_debug("演义摘录：query 与地名为空，跳过")
        return []
    chunks, path_note = _load_all_chunks()
    if not chunks:
        return []

    q_tokens = _tokens(q)
    scored: list[tuple[float, dict[str, Any]]] = []
    for ch in chunks:
        text = str(ch.get("text", "") or "")
        sc = _score_chunk(text, q_tokens, names)
        if sc <= 0:
            continue
        scored.append((sc, ch))
    scored.sort(key=lambda x: (-x[0], str(x[1].get("chunk_id", ""))))
    fk = max(1, top_k)
    head = scored[:fk] if scored else []

    out: list[dict[str, Any]] = []
    if not head:
        _rag_debug(
            "ℹ️ 演义摘录：词法未命中（query 与地名在 chunk 正文中无足够重叠），不附加摘录"
        )
        return []
    for sc, ch in head:
        text = str(ch.get("text", "") or "")
        ex = text if len(text) <= excerpt_max_chars else text[: excerpt_max_chars] + "…"
        out.append(
            {
                "chunk_id": ch.get("chunk_id", ""),
                "chapter_title": ch.get("chapter_title", ""),
                "excerpt": ex,
                "_lex_score": round(sc, 3),
            }
        )
    titles = "、".join(str(x.get("chapter_title", ""))[:24] for x in out) or "（无）"
    _rag_debug(
        f"✅ 演义摘录：route=lexical（关键词+地名命中 jsonl） top_k={len(out)} "
        f"path={path_note} chapters≈{titles}"
    )
    return out
