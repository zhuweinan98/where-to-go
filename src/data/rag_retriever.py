"""轻量 RAG 检索（语义优先，词法回退）。

默认优先使用 embedding 进行语义检索（余弦相似度）；
当 embedding 未配置/不可用时，自动回退到关键词检索，保证可用性。
"""

from __future__ import annotations

import math
import os
import re
from functools import lru_cache
from typing import Any

from openai import OpenAI

from src.data.mock import places_for_city
from src.data.sanguo_kb import SANGUO_PLACES

_TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")
_LAST_RAG_DEBUG_LOGS: list[str] = []


def _rag_debug(msg: str) -> None:
    _LAST_RAG_DEBUG_LOGS.append(msg)
    if os.getenv("LLM_DEBUG", "").strip().lower() in ("1", "true", "yes"):
        print(f"[rag] {msg}", flush=True)


def consume_rag_debug_logs() -> list[str]:
    out = list(_LAST_RAG_DEBUG_LOGS)
    _LAST_RAG_DEBUG_LOGS.clear()
    return out


def _tokens(text: str) -> list[str]:
    t = (text or "").strip().lower()
    if not t:
        return []
    base = _TOKEN_RE.findall(t)
    out: list[str] = []
    for x in base:
        if x and x not in out:
            out.append(x)
    if t not in out:
        out.append(t)
    return out


def _place_text(p: dict[str, Any]) -> str:
    sw = ",".join(str(x) for x in (p.get("suitable_weather") or []))
    return " | ".join(
        [
            str(p.get("city", "")),
            str(p.get("name", "")),
            str(p.get("type", "")),
            str(p.get("reason", "")),
            sw,
        ]
    ).lower()


def _score_place(p: dict[str, Any], query: str) -> float:
    q = (query or "").strip().lower()
    if not q:
        return 0.0
    text = _place_text(p)
    name = str(p.get("name", "")).lower()
    score = 0.0
    if q in text:
        score += 6.0
    for tk in _tokens(q):
        if tk in text:
            score += 2.0
        if tk and tk in name:
            score += 1.0
    return score


def _normalize_v1_base(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if not u:
        return ""
    return u if u.endswith("/v1") else f"{u}/v1"


def _embedding_client_info() -> dict[str, Any] | None:
    mode = os.getenv("LLM_MODE", "off").strip().lower()
    force = os.getenv("RAG_EMBEDDING_MODE", "auto").strip().lower()
    if force == "off":
        _rag_debug("embedding 已关闭（RAG_EMBEDDING_MODE=off），将回退词法检索")
        return None
    if mode == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            _rag_debug("未配置 OPENAI_API_KEY，语义检索不可用，将回退词法检索")
            return None
        base = _normalize_v1_base(
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
        )
        model = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small").strip()
        timeout = float(os.getenv("RAG_EMBEDDING_TIMEOUT", "30").strip() or "30")
        return {"api_key": key, "base_url": base, "model": model, "timeout": timeout}
    if mode == "ollama":
        base = _normalize_v1_base(
            os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        )
        key = os.getenv("OLLAMA_API_KEY", "ollama").strip() or "ollama"
        model = os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text").strip()
        timeout = float(os.getenv("RAG_EMBEDDING_TIMEOUT", "30").strip() or "30")
        return {"api_key": key, "base_url": base, "model": model, "timeout": timeout}
    _rag_debug(f"当前 LLM_MODE={mode!r} 不支持 embedding，回退词法检索")
    return None


@lru_cache(maxsize=2)
def _get_embedding_client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def _embed_texts(texts: list[str]) -> list[list[float]] | None:
    info = _embedding_client_info()
    if info is None:
        return None
    try:
        _rag_debug(
            f"尝试 embedding：model={info['model']!r} base_url={info['base_url']} "
            f"texts={len(texts)}"
        )
        client = _get_embedding_client(
            info["base_url"], info["api_key"], float(info["timeout"])
        )
        res = client.embeddings.create(model=info["model"], input=texts)
        out: list[list[float]] = []
        for d in res.data:
            out.append([float(x) for x in d.embedding])
        dim = len(out[0]) if out else 0
        _rag_debug(
            f"embedding 成功：model={info['model']!r} vectors={len(out)} dim={dim}"
        )
        return out
    except Exception as e:
        # embedding 不可用时自动回退词法检索，避免影响主链路可用性
        _rag_debug(
            f"embedding 失败：model={info['model']!r} error={e}，将回退词法检索。"
            "若为 Ollama 请先 `ollama pull <embedding模型>`。"
        )
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        x = a[i]
        y = b[i]
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return -1.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _semantic_rank(
    docs: list[dict[str, Any]],
    query: str,
    *,
    top_k: int,
) -> list[dict[str, Any]] | None:
    if not query.strip() or not docs:
        return []
    doc_texts = [_place_text(x) for x in docs]
    vectors = _embed_texts([query] + doc_texts)
    if vectors is None or len(vectors) != len(docs) + 1:
        return None
    qv = vectors[0]
    scored: list[tuple[float, dict[str, Any]]] = []
    for i, p in enumerate(docs):
        sim = _cosine(qv, vectors[i + 1])
        scored.append((sim, p))
    scored.sort(key=lambda x: (-x[0], str(x[1].get("name", ""))))
    pick = scored[: max(1, top_k)]
    out: list[dict[str, Any]] = []
    info = _embedding_client_info()
    model_name = str(info.get("model", "")) if info is not None else ""
    for sim, p in pick:
        item = dict(p)
        item["_score"] = float(sim)
        item["_score_type"] = "semantic_cosine"
        item["_retrieval_engine"] = "semantic"
        if model_name:
            item["_embedding_model"] = model_name
        out.append(item)
    return out


def _lexical_rank(
    docs: list[dict[str, Any]],
    query: str,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return [dict(p) for p in docs]

    scored: list[tuple[float, dict[str, Any]]] = []
    for p in docs:
        s = _score_place(p, q)
        scored.append((s, p))
    scored.sort(key=lambda x: (-x[0], str(x[1].get("name", ""))))

    hits = [(s, p) for s, p in scored if s > 0]
    pick = hits[: max(1, top_k)] if hits else scored[: max(1, top_k)]
    out: list[dict[str, Any]] = []
    for s, p in pick:
        item = dict(p)
        item["_score"] = float(s)
        item["_score_type"] = "lexical_rule"
        item["_retrieval_engine"] = "lexical_fallback"
        if not hits:
            item["_retrieval_fallback"] = True
        out.append(item)
    return out


def _rank_docs(
    docs: list[dict[str, Any]],
    query: str,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    semantic = _semantic_rank(docs, query, top_k=top_k)
    if semantic is not None:
        return semantic
    return _lexical_rank(docs, query, top_k=top_k)


def search_places(city: str, query: str, *, top_k: int = 3) -> list[dict[str, Any]]:
    _LAST_RAG_DEBUG_LOGS.clear()
    local = places_for_city(city)
    q = (query or "").strip()
    if not q:
        return [dict(p) for p in local]
    return _rank_docs(local, q, top_k=top_k)


def search_sanguo_places(query: str, *, top_k: int = 3) -> list[dict[str, Any]]:
    _LAST_RAG_DEBUG_LOGS.clear()
    q = (query or "").strip()
    if not q:
        return [dict(x) for x in SANGUO_PLACES[: max(1, top_k)]]
    return _rank_docs(SANGUO_PLACES, q, top_k=top_k)
