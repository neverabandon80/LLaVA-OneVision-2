"""统计 tar 包计数。

支持解析如下行格式：
  pretrain-000592.tar: 618

输出：
1) 每个 tar 对应的数值
2) tar 包总数
3) 所有 tar 数值求和
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


LINE_PATTERN = re.compile(r"^\s*(?P<name>[^:#\s]+\.tar)\s*:\s*(?P<count>\d+)\s*$")


def parse_tar_counts(file_path: Path) -> list[tuple[str, int]]:
    """从文件中提取 (tar_name, count) 列表。"""
    results: list[tuple[str, int]] = []
    with file_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            match = LINE_PATTERN.match(raw_line)
            if match is None:
                continue
            tar_name = match.group("name")
            count_value = int(match.group("count"))
            results.append((tar_name, count_value))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="统计每个 tar 的数量和总数量")
    parser.add_argument("input", type=Path, help="输入文件路径，例如 .info.yaml")
    parser.add_argument("--output", type=Path, default=None, help="可选：将详细结果写入文件")
    parser.add_argument("--sort-by-name", action="store_true", help="按 tar 名称排序输出（默认按文件出现顺序）")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    tar_counts = parse_tar_counts(args.input)
    if len(tar_counts) == 0:
        raise ValueError("未找到符合 '<xxx>.tar: <数字>' 格式的行")

    if args.sort_by_name:
        tar_counts = sorted(tar_counts, key=lambda item: item[0])

    total_tar_files = len(tar_counts)
    total_count_sum = sum(item[1] for item in tar_counts)

    lines: list[str] = []
    lines.append(f"input_file: {args.input}")
    lines.append(f"tar_files: {total_tar_files}")
    lines.append(f"total_count_sum: {total_count_sum}")
    lines.append("details:")
    for tar_name, count_value in tar_counts:
        lines.append(f"  {tar_name}: {count_value}")

    output_text = "\n".join(lines)
    print(output_text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
