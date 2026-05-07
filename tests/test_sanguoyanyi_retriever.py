"""演义 jsonl 词法摘录（与三国地点同路径，单测不请求外网）。"""

from __future__ import annotations

import pytest

from src.data.sanguoyanyi_retriever import _load_all_chunks, search_romance_excerpts


@pytest.fixture(autouse=True)
def clear_chunk_cache():
    _load_all_chunks.cache_clear()
    yield
    _load_all_chunks.cache_clear()


def test_romance_excerpts_chibi_query_non_empty():
    ex = search_romance_excerpts("赤壁 曹操", ["赤壁古战场"], top_k=2)
    assert isinstance(ex, list)
    assert len(ex) >= 1
    blob = " ".join(
        f"{x.get('chapter_title', '')} {x.get('excerpt', '')}" for x in ex
    )
    assert "赤壁" in blob or "曹操" in blob
    assert all(x.get("chunk_id") for x in ex)
