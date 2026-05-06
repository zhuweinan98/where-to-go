"""三国演义按回切块（不依赖大部头原文文件）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.sanguoyanyi_chunking import (
    build_chunks_from_file,
    chapters_to_chunk_dicts,
    split_into_chapters,
)


SAMPLE = """
序前空行可忽略

第一回 宴桃园豪杰三结义 斩黄巾英雄首立功

临江仙词一首。

话说天下大势。

第二回 张翼德怒鞭督邮 何国舅谋诛宦竖

且说董卓。
"""


def test_split_two_chapters_and_preamble_merged():
    ch = split_into_chapters(SAMPLE)
    assert len(ch) == 2
    assert ch[0][0].startswith("第一回")
    assert "序前空行" in ch[0][1] or "临江仙" in ch[0][1]
    assert "临江仙" in ch[0][1]
    assert "话说天下大势" in ch[0][1]
    assert ch[1][0].startswith("第二回")
    assert "董卓" in ch[1][1]


def test_chunks_dict_shape():
    ch = split_into_chapters(SAMPLE)
    rows = chapters_to_chunk_dicts(ch, doc_id="test_doc")
    assert len(rows) == 2
    assert rows[0]["chunk_id"] == "test_doc_hui_0001"
    assert rows[0]["chapter_seq"] == 1
    assert rows[1]["chapter_seq"] == 2
    assert "text" in rows[0] and len(rows[0]["text"]) > 0


def test_full_raw_file_if_present():
    root = Path(__file__).resolve().parent.parent
    raw = root / "data" / "raw" / "三国演义（原文版）.txt"
    if not raw.is_file():
        pytest.skip("未检出大部头原文，跳过集成条数断言")
    chunks = build_chunks_from_file(raw)
    # 常见版本 120 回；若版本不同仅要求「多条且每条有 text」
    assert len(chunks) >= 100
    assert all(c.get("chapter_title", "").startswith("第") for c in chunks)
    assert all(len(c.get("text", "")) > 50 for c in chunks[:5])
    line = json.dumps(chunks[0], ensure_ascii=False)
    assert "chunk_id" in line and "chapter_title" in line


def test_chapter_line_regex_accepts_一百零一():
    text = "第一百零一回 出陇上诸葛妆神 奔剑阁张郃中计\n\n正文。\n"
    ch = split_into_chapters(text)
    assert len(ch) == 1
    assert ch[0][0].startswith("第一百零一回")
    assert "正文" in ch[0][1]
