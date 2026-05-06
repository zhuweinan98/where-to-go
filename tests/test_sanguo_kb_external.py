"""三国知识库外置 JSON：默认路径与 SANGUO_KB_JSON 覆盖（子进程隔离 import）。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.data.sanguo_kb import SANGUO_PLACES, _default_json_path


def test_default_json_file_exists():
    p = _default_json_path()
    assert p.is_file(), f"缺少外置知识库文件: {p}"


def test_sanguo_places_matches_bundled_json():
    assert len(SANGUO_PLACES) == 5
    names = {x["name"] for x in SANGUO_PLACES}
    assert "许昌曹魏故城遗址" in names
    raw = json.loads(_default_json_path().read_text(encoding="utf-8"))
    assert len(raw) == len(SANGUO_PLACES)


@pytest.mark.parametrize(
    "tags_field",
    [p["tags"] for p in SANGUO_PLACES],
)
def test_each_place_has_list_tags(tags_field):
    assert isinstance(tags_field, list)


def test_sanguo_kb_json_env_overrides_in_subprocess():
    """子进程内首次 import，避免与已加载的 SANGUO_PLACES 冲突。"""
    root = Path(__file__).resolve().parent.parent
    one = [
        {
            "name": "外置单条测",
            "modern_city": "测市",
            "province": "测省",
            "era_role": "测",
            "tags": ["测"],
            "visit_hint": "测",
        }
    ]
    payload = json.dumps(one, ensure_ascii=False)
    script = f"""
import json, os, tempfile
from pathlib import Path
d = Path(tempfile.mkdtemp())
p = d / "kb.json"
p.write_text({repr(payload)}, encoding="utf-8")
os.environ["SANGUO_KB_JSON"] = str(p)
import sys
sys.path.insert(0, {json.dumps(str(root))})
from src.data import sanguo_kb as m
assert len(m.SANGUO_PLACES) == 1
assert m.SANGUO_PLACES[0]["name"] == "外置单条测"
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(root),
        check=True,
        capture_output=True,
        text=True,
    )
