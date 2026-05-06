"""轻量 RAG 检索（语义优先，词法回退）。

默认优先使用 embedding 进行语义检索（余弦相似度）；
当 embedding 未配置/不可用时，自动回退到关键词检索，保证可用性。

Mock 景点与三国知识库字段结构不同，使用各自的「文档拼接」函数参与向量与词法匹配。

两阶段召回：先以较大的 retrieve_k 做 _rank_docs，再在命中集上做轻量规则重排，截断为最终 top_k。
当候选数或 RAG_RETRIEVE_K 不大于最终 top_k 时，自动退化为单阶段（与仅 _rank_docs 等价）。
环境变量 RAG_RETRIEVE_K 控制第一阶段宽度（默认 12）。
"""

from __future__ import annotations

import math
import os
import re
from functools import lru_cache
from typing import Any, Callable

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


def _mock_place_doc_text(p: dict[str, Any]) -> str:
    """Mock 景点表结构：city / name / type / reason / suitable_weather。"""
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


def _sanguo_doc_text(p: dict[str, Any]) -> str:
    """三国知识库结构：name / 省区市 / 史实角色 / 标签 / 游玩建议。"""
    raw_tags = p.get("tags")
    if isinstance(raw_tags, list):
        tags_s = ",".join(str(x) for x in raw_tags)
    else:
        tags_s = str(raw_tags or "")
    return " | ".join(
        [
            str(p.get("name", "")),
            str(p.get("province", "")),
            str(p.get("modern_city", "")),
            str(p.get("era_role", "")),
            tags_s,
            str(p.get("visit_hint", "")),
        ]
    ).lower()


def _score_place(
    p: dict[str, Any],
    query: str,
    *,
    doc_text_fn: Callable[[dict[str, Any]], str],
) -> float:
    q = (query or "").strip().lower()
    if not q:
        return 0.0
    text = doc_text_fn(p)
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
    if mode == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            _rag_debug("语义检索：未配置云端密钥，已改用关键词检索。")
            return None
        base = _normalize_v1_base(
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
        )
        model = os.getenv("RAG_EMBEDDING_MODEL", "deepseek-embedding").strip()
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
    _rag_debug(
        f"语义检索：当前是「{mode}」模式，不走向量，已改用关键词检索。"
    )
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
            f"语义检索·embedding：调用 embeddings API，model={info['model']}，"
            f"batch={len(texts)}（1 段 query + 各条目）。"
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
            f"语义检索·embedding：完成，dim={dim}，下一步 cosine 相似度排序。"
        )
        return out
    except Exception as e:
        # embedding 不可用时自动回退词法检索，避免影响主链路可用性
        _rag_debug(
            "语义检索：向量接口失败，已自动改用「关键词」方式排序。"
            f"（原因：{str(e)[:200]}；若用 Ollama 请先拉取 embedding 模型。）"
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
    doc_text_fn: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]] | None:
    if not query.strip() or not docs:
        return []
    doc_texts = [doc_text_fn(x) for x in docs]
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
    doc_text_fn: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return [dict(p) for p in docs]

    scored: list[tuple[float, dict[str, Any]]] = []
    for p in docs:
        s = _score_place(p, q, doc_text_fn=doc_text_fn)
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
    doc_text_fn: Callable[[dict[str, Any]], str] = _mock_place_doc_text,
) -> list[dict[str, Any]]:
    semantic = _semantic_rank(
        docs, query, top_k=top_k, doc_text_fn=doc_text_fn
    )
    if semantic is not None:
        return semantic
    return _lexical_rank(docs, query, top_k=top_k, doc_text_fn=doc_text_fn)


def _rag_retrieve_k(default: int = 12) -> int:
    raw = os.getenv("RAG_RETRIEVE_K", "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _mock_rerank_bonus(p: dict[str, Any], query: str) -> float:
    """同城景点：按用户 query 关键词与结构化字段对齐加分（无网络）。"""
    q = (query or "").strip().lower()
    if not q:
        return 0.0
    bonus = 0.0
    t = str(p.get("type", "")).lower()
    r = str(p.get("reason", "")).lower()
    sw = p.get("suitable_weather") or []
    sws = (
        ",".join(str(x) for x in sw).lower()
        if isinstance(sw, list)
        else str(sw).lower()
    )
    price = p.get("price")
    try:
        ip = int(price) if price is not None else -1
    except (TypeError, ValueError):
        ip = -1

    if any(k in q for k in ("免费", "不花钱", "零元")) and ip == 0:
        bonus += 2.0
    if any(k in q for k in ("室内", "下雨", "雨天")) and (
        "室内" in r
        or "雨天" in r
        or "雨" in sws
        or any(x in t for x in ("博物馆", "美术馆", "博物院"))
    ):
        bonus += 2.0
    if ("夜景" in q or "晚上" in q) and "夜景" in r:
        bonus += 1.0
    if any(k in q for k in ("江", "滨江", "沿江", "江边")) and (
        "江" in r or "滨" in r or "沿江" in r
    ):
        bonus += 1.0
    return bonus


def _sanguo_rerank_bonus(p: dict[str, Any], query: str) -> float:
    """三国条目：query 分词与史实/标签/建议文本重合加分。"""
    q = (query or "").strip().lower()
    if not q:
        return 0.0
    era = str(p.get("era_role", "")).lower()
    hint = str(p.get("visit_hint", "")).lower()
    raw_tags = p.get("tags")
    if isinstance(raw_tags, list):
        tags_s = ",".join(str(x) for x in raw_tags).lower()
    else:
        tags_s = str(raw_tags or "").lower()
    blob = f"{era} {hint} {tags_s}"
    bonus = 0.0
    for tk in _tokens(q):
        if len(tk) >= 2 and tk in blob:
            bonus += 0.45
    return min(bonus, 3.0)


def _first_retrieval_route(hits: list[dict[str, Any]]) -> str:
    """第一条命中的检索引擎（semantic / lexical_fallback）。"""
    if not hits:
        return "-"
    return str(hits[0].get("_retrieval_engine", "-"))


def _hits_name_flow(hits: list[dict[str, Any]], *, limit: int = 24) -> str:
    """仅地点名，用箭头表示第一轮相关度顺序（便于人读）。"""
    if not hits:
        return "（无）"
    head = [str(x.get("name", "?")) for x in hits[:limit]]
    tail = ""
    if len(hits) > limit:
        tail = f" …（其余 {len(hits) - limit} 条略）"
    return " → ".join(head) + tail


def _hits_name_join(hits: list[dict[str, Any]], *, limit: int = 12) -> str:
    """顿号连接，用于最终名单一行话。"""
    if not hits:
        return "（无）"
    names = [str(x.get("name", "?")) for x in hits[:limit]]
    if len(hits) > limit:
        names.append(f"等{len(hits)}条")
    return "、".join(names)


def _lightweight_rerank(
    hits: list[dict[str, Any]],
    query: str,
    *,
    final_k: int,
    bonus_fn: Callable[[dict[str, Any], str], float],
) -> list[dict[str, Any]]:
    """对第一阶段命中列表做 min-max(_score)+bonus 重排，截断为 final_k。"""
    if not hits:
        return []
    fk = max(1, final_k)
    raw_scores = [float(x.get("_score", 0.0)) for x in hits]
    norms = _min_max_normalize(raw_scores)
    scored: list[tuple[float, dict[str, Any]]] = []
    for h, base in zip(hits, norms):
        p = dict(h)
        b = bonus_fn(p, query)
        rr = float(base) + float(b)
        p["_rerank_base_norm"] = float(base)
        p["_rerank_bonus"] = float(b)
        p["_rerank_score"] = rr
        p["_rerank_stage"] = "lightweight_v1"
        scored.append((rr, p))
    scored.sort(key=lambda x: (-x[0], str(x[1].get("name", ""))))
    out = [p for _, p in scored[:fk]]
    _rag_debug(
        f"✅ rerank（轻量重排）：n_in={len(hits)} final_k={fk} "
        f"names={_hits_name_join(out)}（base=minmax(_score)+bonus）"
    )
    return out


def search_places(city: str, query: str, *, top_k: int = 3) -> list[dict[str, Any]]:
    _LAST_RAG_DEBUG_LOGS.clear()
    local = places_for_city(city)
    q = (query or "").strip()
    if not q:
        return [dict(p) for p in local]

    fk = max(1, top_k)
    rk = _rag_retrieve_k(12)
    pool_n = len(local)
    if rk <= fk:
        out = _rank_docs(local, q, top_k=fk)
        _rag_debug(
            f"✅【{city}】单阶段检索：pool={pool_n} top_k={fk} route={_first_retrieval_route(out)} "
            f"order={_hits_name_flow(out)}"
        )
        return out

    rk_eff = min(max(rk, fk), pool_n)
    stage1 = _rank_docs(
        local, q, top_k=rk_eff, doc_text_fn=_mock_place_doc_text
    )
    _rag_debug(
        f"✅【{city}】第一轮 recall：pool={pool_n} stage1_top_k={rk_eff} route={_first_retrieval_route(stage1)} "
        f"order={_hits_name_flow(stage1)}"
    )
    return _lightweight_rerank(
        stage1, q, final_k=fk, bonus_fn=_mock_rerank_bonus
    )


def search_sanguo_places(query: str, *, top_k: int = 3) -> list[dict[str, Any]]:
    _LAST_RAG_DEBUG_LOGS.clear()
    q = (query or "").strip()
    if not q:
        return [dict(x) for x in SANGUO_PLACES[: max(1, top_k)]]

    fk = max(1, top_k)
    rk = _rag_retrieve_k(12)
    n_docs = len(SANGUO_PLACES)
    if rk <= fk:
        out = _rank_docs(
            SANGUO_PLACES, q, top_k=fk, doc_text_fn=_sanguo_doc_text
        )
        _rag_debug(
            f"✅【三国】单阶段检索：pool={n_docs} top_k={fk} route={_first_retrieval_route(out)} "
            f"order={_hits_name_flow(out)}"
        )
        return out

    rk_eff = min(max(rk, fk), n_docs)
    stage1 = _rank_docs(
        SANGUO_PLACES, q, top_k=rk_eff, doc_text_fn=_sanguo_doc_text
    )
    _rag_debug(
        f"✅【三国】第一轮 recall：pool={n_docs} stage1_top_k={rk_eff} route={_first_retrieval_route(stage1)} "
        f"order={_hits_name_flow(stage1)}"
    )
    return _lightweight_rerank(
        stage1, q, final_k=fk, bonus_fn=_sanguo_rerank_bonus
    )
