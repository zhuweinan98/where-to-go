#!/usr/bin/env python3
"""从 data/raw/三国演义（原文版）.txt 按「回」切块，写入 data/kb/sanguoyanyi_chunks.jsonl。

用法（在项目根目录）:
  python3 scripts/build_sanguoyanyi_chunks.py
  python3 scripts/build_sanguoyanyi_chunks.py --input path/to.txt --output path/to.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 保证可从仓库根 import src.*
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.sanguoyanyi_chunking import (  # noqa: E402
    build_chunks_from_file,
    default_jsonl_path,
    default_raw_txt_path,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="三国演义原文 → jsonl chunks")
    ap.add_argument(
        "--input",
        type=Path,
        default=default_raw_txt_path(),
        help="原文 txt 路径",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=default_jsonl_path(),
        help="输出 jsonl 路径",
    )
    ap.add_argument(
        "--doc-id",
        default="sanguoyanyi_v1",
        help="写入每条记录的 doc_id",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"错误：找不到输入文件: {args.input.resolve()}", file=sys.stderr)
        return 1

    chunks = build_chunks_from_file(args.input, doc_id=args.doc_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"✅ 切块完成：chunks={len(chunks)} input={args.input} output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
