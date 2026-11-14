"""YAML loader with PyYAML support and a zero-dependency pure-Python fallback.

Allows the offline pipeline to run out-of-the-box in standard Python 3.10+
even before `pip install -r pipeline/requirements.txt` is executed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if token in ("null", "Null", "NULL", "~", ""):
        return None
    if token in ("true", "True", "TRUE"):
        return True
    if token in ("false", "False", "FALSE"):
        return False
    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
        return token[1:-1]
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    try:
        if "." in token or "e" in token.lower():
            return float(token)
        return int(token)
    except ValueError:
        return token


def _strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if idx == 0 or line[idx - 1].isspace():
                return line[:idx].rstrip()
    return line.rstrip()


def _fallback_parse_yaml(text: str) -> Any:
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        cleaned = _strip_inline_comment(raw)
        if not cleaned.strip():
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        lines.append((indent, cleaned.strip()))

    def parse_block(index: int, expected_indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index
        _, first = lines[index]
        if first.startswith("- "):
            items: list[Any] = []
            while index < len(lines):
                indent, content = lines[index]
                if indent < expected_indent or not content.startswith("- "):
                    break
                if indent > expected_indent:
                    break
                rest = content[2:].strip()
                if ":" in rest and not rest.startswith('"') and not rest.startswith("'"):
                    key, val_part = rest.split(":", 1)
                    obj: dict[str, Any] = {}
                    key = key.strip()
                    val_part = val_part.strip()
                    index += 1
                    if val_part:
                        obj[key] = _parse_scalar(val_part)
                    else:
                        if index < len(lines) and lines[index][0] > indent:
                            child, index = parse_block(index, lines[index][0])
                            obj[key] = child
                        else:
                            obj[key] = None
                    # Consume sibling keys belonging to this list item's dict
                    while index < len(lines) and lines[index][0] > indent:
                        child_indent, child_content = lines[index]
                        if child_content.startswith("- "):
                            break
                        ckey, cval = child_content.split(":", 1)
                        ckey = ckey.strip()
                        cval = cval.strip()
                        index += 1
                        if cval:
                            obj[ckey] = _parse_scalar(cval)
                        elif index < len(lines) and lines[index][0] > child_indent:
                            sub, index = parse_block(index, lines[index][0])
                            obj[ckey] = sub
                        else:
                            obj[ckey] = None
                    items.append(obj)
                else:
                    items.append(_parse_scalar(rest))
                    index += 1
            return items, index

        mapping: dict[str, Any] = {}
        while index < len(lines):
            indent, content = lines[index]
            if indent < expected_indent:
                break
            if content.startswith("- "):
                break
            key, val_part = content.split(":", 1)
            key = key.strip()
            val_part = val_part.strip()
            index += 1
            if val_part:
                mapping[key] = _parse_scalar(val_part)
            elif index < len(lines) and lines[index][0] > indent:
                child, index = parse_block(index, lines[index][0])
                mapping[key] = child
            else:
                mapping[key] = None
        return mapping, index

    parsed, _ = parse_block(0, lines[0][0] if lines else 0)
    return parsed


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file using PyYAML when installed or the built-in subset parser."""
    content = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(content)
        return data if isinstance(data, dict) else {}
    except ImportError:
        data = _fallback_parse_yaml(content)
        return data if isinstance(data, dict) else {}
