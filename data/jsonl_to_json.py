"""
jsonl_to_json.py - Convert JSONL into structured JSON with a tiny ASCII CLI.

Usage:
    python jsonl_to_json.py input.jsonl -o output.json
    python jsonl_to_json.py input.jsonl -o output.json --array
    python jsonl_to_json.py input.jsonl -o output.json --skip-invalid
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


BANNER = r"""
+--------------------------------------------------------------+
|  _ ____   ___  _   _ _      ____       _ ____   ___  _   _   |
| | / ___| / _ \| \ | | |    |___ \     | / ___| / _ \| \ | |  |
| | \___ \| | | |  \| | |_____ __) | _  | \___ \| | | |  \| |  |
| | |___) | |_| | |\  | |_____/ __/ | |_| |___) | |_| | |\  |  |
| |_|____/ \___/|_| \_|_|    |_____(_)___/|____/ \___/|_| \_|  |
|                                                              |
|              JSONL -> structured JSON transformer             |
+--------------------------------------------------------------+
"""


def positive_indent(value: str) -> int:
    try:
        indent = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("indent must be an integer") from exc
    if indent < 0:
        raise argparse.ArgumentTypeError("indent must be 0 or greater")
    return indent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform newline-delimited JSON into structured JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Path to the source .jsonl file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to write the .json file. Defaults to input name with .json suffix.",
    )
    parser.add_argument(
        "--array",
        action="store_true",
        help="Write a bare JSON array instead of an object with metadata.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip invalid JSONL rows instead of failing on the first error.",
    )
    parser.add_argument(
        "--indent",
        type=positive_indent,
        default=2,
        help="Spaces to use when pretty-printing JSON.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON with no extra whitespace.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the ASCII banner and summary.",
    )
    return parser.parse_args()


def read_jsonl(path: Path, skip_invalid: bool) -> tuple[list[object], list[dict[str, str]]]:
    records = []
    errors = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                error = {
                    "line": str(line_number),
                    "column": str(exc.colno),
                    "message": exc.msg,
                }
                errors.append(error)
                if not skip_invalid:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}, column {exc.colno}: {exc.msg}"
                    ) from exc

    return records, errors


def build_payload(source: Path, records: list[object], errors: list[dict[str, str]]) -> dict[str, object]:
    return {
        "metadata": {
            "source": str(source),
            "record_count": len(records),
            "invalid_rows_skipped": len(errors),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "records": records,
    }


def write_json(path: Path, payload: object, indent: int, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if compact:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(payload, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")


def print_summary(input_path: Path, output_path: Path, record_count: int, errors: list[dict[str, str]]) -> None:
    skipped = len(errors)
    print(BANNER)
    print("+---------------- conversion summary ----------------+")
    print(f"| input   : {input_path}")
    print(f"| output  : {output_path}")
    print(f"| records : {record_count}")
    print(f"| skipped : {skipped}")
    print("+----------------------------------------------------+")
    if skipped:
        print("\nSkipped invalid rows:")
        for error in errors[:10]:
            print(f"  - line {error['line']}, col {error['column']}: {error['message']}")
        if skipped > 10:
            print(f"  ... and {skipped - 10} more")


def main() -> int:
    args = parse_args()
    input_path = args.input
    output_path = args.output or input_path.with_suffix(".json")

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    if input_path.is_dir():
        raise SystemExit(f"Input path is a directory: {input_path}")

    try:
        records, errors = read_jsonl(input_path, args.skip_invalid)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    payload = records if args.array else build_payload(input_path, records, errors)
    write_json(output_path, payload, args.indent, args.compact)

    if not args.quiet:
        print_summary(input_path, output_path, len(records), errors)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
