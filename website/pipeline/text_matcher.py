"""Italian phrase normalization and closed-set dictionary matching.

Uses rapidfuzz when available and falls back to difflib + token overlap + character n-grams.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


def normalize_italian(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower().strip())
    no_accents = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^a-z0-9\s]", " ", no_accents)
    return re.sub(r"\s+", " ", cleaned).strip()


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    try:
        from rapidfuzz import fuzz  # type: ignore

        return max(
            fuzz.ratio(a, b) / 100.0,
            fuzz.token_sort_ratio(a, b) / 100.0,
        )
    except ImportError:
        seq = SequenceMatcher(None, a, b).ratio()
        ta, tb = set(a.split()), set(b.split())
        jacc = len(ta & tb) / max(1, len(ta | tb))
        return max(seq, 0.55 * jacc + 0.45 * seq)


@dataclass
class MatchCandidate:
    class_id: str
    phrase: str
    matched_text: str
    score: float
    via_synonym: bool


@dataclass
class MatchResult:
    matched: bool
    best: MatchCandidate | None
    suggestions: list[MatchCandidate]
    normalized_query: str


def match_phrase(query: str, dict_cfg: dict[str, Any]) -> MatchResult:
    norm_q = normalize_italian(query)
    threshold = float(dict_cfg.get("metadata", {}).get("match_threshold", 0.52))
    max_sugg = int(dict_cfg.get("metadata", {}).get("max_suggestions", 3))
    if not norm_q:
        return MatchResult(False, None, [], norm_q)

    candidates: list[MatchCandidate] = []
    for cls in dict_cfg.get("classes", []):
        best_score = _similarity(norm_q, normalize_italian(cls["phrase"]))
        best_txt = cls["phrase"]
        via_syn = False
        for syn in cls.get("synonyms", []):
            s_score = _similarity(norm_q, normalize_italian(syn))
            if s_score > best_score:
                best_score = s_score
                best_txt = syn
                via_syn = True
        candidates.append(
            MatchCandidate(
                class_id=cls["id"],
                phrase=cls["phrase"],
                matched_text=best_txt,
                score=round(best_score, 4),
                via_synonym=via_syn,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    top = candidates[0] if candidates else None
    is_match = top is not None and top.score >= threshold
    return MatchResult(
        matched=is_match,
        best=top if is_match else None,
        suggestions=candidates[:max_sugg],
        normalized_query=norm_q,
    )
