"""invoke 前 SystemMessage 快照：结构单测（不请求模型）。"""

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.langchain_fc import _lc_message_role, _system_message_blocks


def test_lc_message_role_system_human():
    assert _lc_message_role(SystemMessage(content="x")) == "system"
    assert _lc_message_role(HumanMessage(content="y")) == "human"


def test_system_message_blocks_slots_and_preview():
    msgs = [
        SystemMessage(content="fc rules"),
        SystemMessage(content="rag block"),
        HumanMessage(content="用户问"),
    ]
    blocks = _system_message_blocks(msgs)
    assert [b["slot"] for b in blocks] == ["fc_system", "sanguo_rag_system"]
    assert blocks[0]["chars"] == len("fc rules")
    assert "fc rules" in blocks[0]["preview"]
    assert blocks[0].get("full_text") == "fc rules"
    assert "rag block" in blocks[1]["preview"]
    assert blocks[1].get("full_text") == "rag block"
