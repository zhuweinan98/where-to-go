"""两阶段召回 + 轻量重排：元数据与开关行为。"""

from __future__ import annotations

import pytest

from src.data.rag_retriever import search_places


def test_two_stage_on_adds_rerank_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_RETRIEVE_K", "12")
    hits = search_places("北京", "博物馆", top_k=2)
    assert len(hits) <= 2
    assert hits
    assert "_rerank_score" in hits[0]
    assert hits[0].get("_rerank_stage") == "lightweight_v1"


def test_retrieve_k_le_final_skips_two_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_RETRIEVE_K", "3")
    hits = search_places("广州", "美术馆", top_k=3)
    assert hits
    assert "_rerank_score" not in hits[0]
