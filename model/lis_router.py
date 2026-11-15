"""
lis_router.py - from Italian text to the paths of the LIS animations.

Flow:
  1. best semantic match (sentence encoder) between the input and the dataset sentences
     - score >= tau, margin over the second candidate, guards on negation and numbers
     -> mode="sentence", paths=[sample folder]
  2. otherwise: spaCy (POS + lemma) -> reorder into LIS order -> look up the
     single words in the dataset (missing words are skipped)
     -> mode="words", paths=[folders in order], modifiers=[...]
  3. no word found -> mode="none"

Expected dataset layout (as in the sample):
  <root>/<any subfolder>/<sample_id>/metadata.json   ("sentence" field)
                                   /left_hand.csv, right_hand.csv, face.csv, body.csv
The returned "animation path" is the sample folder.

Single-word clips are the samples whose "sentence" is one word.
Optional overrides in metadata.json: "type": "word" | "sentence", "lemma": "<lemma>".

Dependencies:
  pip install sentence-transformers spacy numpy
  python -m spacy download it_core_news_sm      # _md or _lg give better lemmas

Usage:
  python lis_router.py query DATASET_ROOT "il treno parte alle 19"
  python lis_router.py calibrate DATASET_ROOT calibration.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Italian lexicons (data: keep them in Italian)
# ----------------------------------------------------------------------------
NEG = {"non", "mai", "niente", "nulla", "nessuno", "nessuna", "senza"}
WH = {"chi", "che", "cosa", "dove", "quando", "perché", "perche", "come", "quanto", "quale", "quali"}
TIME = {"ieri", "oggi", "domani", "dopodomani", "adesso", "ora", "stasera", "stamattina",
        "mattina", "pomeriggio", "sera", "notte"}
DROP_POS = {"DET", "AUX", "ADP", "CCONJ", "SCONJ", "PART"}
DROP_LEMMAS = {"essere"}  # LIS has no copula

# No "uno/una/sei": they are also an article / a verb and would give false positives.
NUMBER_WORDS = {
    "zero": 0, "due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sette": 7,
    "otto": 8, "nove": 9, "dieci": 10, "undici": 11, "dodici": 12, "tredici": 13,
    "quattordici": 14, "quindici": 15, "sedici": 16, "diciassette": 17,
    "diciotto": 18, "diciannove": 19, "venti": 20, "trenta": 30,
    "quaranta": 40, "cinquanta": 50,
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(norm: str) -> list[str]:
    return re.findall(r"\w+", norm)


def _numbers(norm: str) -> frozenset[int]:
    out = set()
    for tok in _tokens(norm):
        if tok.isdigit():
            out.add(int(tok))
        elif tok in NUMBER_WORDS:
            out.add(NUMBER_WORDS[tok])
    return frozenset(out)


def _has_neg(norm: str) -> bool:
    return bool(NEG & set(_tokens(norm)))


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Entry:
    path: Path
    sentence: str
    norm: str
    lemma: str | None = None
    kind: str | None = None  # "word" | "sentence" | None (auto)


@dataclass
class Probe:
    idx: int
    score: float
    second: float
    guards_ok: bool


@dataclass
class Result:
    mode: str                                   # "sentence" | "words" | "none"
    paths: list[str] = field(default_factory=list)
    score: float = 0.0                          # best similarity found
    matched_label: str | None = None            # best dataset sentence (also when rejected)
    order: list[str] = field(default_factory=list)      # lemmas in LIS order ("words" only)
    skipped: list[str] = field(default_factory=list)    # lemmas without a clip
    modifiers: list[str] = field(default_factory=list)  # face channels to overlay

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def load_dataset(root: Path) -> list[Entry]:
    entries = []
    for meta in sorted(root.rglob("metadata.json")):
        try:
            d = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sentence = d.get("sentence")
        if not sentence or not normalize(sentence):
            continue
        entries.append(Entry(meta.parent, sentence, normalize(sentence),
                             d.get("lemma"), d.get("type")))
    return entries


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
class LisRouter:
    def __init__(self, dataset_root: str | Path,
                 model: str = "intfloat/multilingual-e5-base",
                 spacy_model: str = "it_core_news_sm",
                 tau: float = 0.88, margin: float = 0.02,
                 min_coverage: float = 0.0):
        """
        tau           cosine threshold to accept a whole-sentence match
        margin        minimum gap between the first and the best *different* candidate
        min_coverage  minimum fraction of words found in the fallback (0 = one is enough)
        """
        import spacy
        from sentence_transformers import SentenceTransformer

        self.root = Path(dataset_root)
        self.tau, self.margin, self.min_coverage = tau, margin, min_coverage
        self.entries = load_dataset(self.root)
        if not self.entries:
            raise ValueError(f"No valid metadata.json under {self.root}")

        self.nlp = spacy.load(spacy_model)
        self.model_name = model
        self.encoder = SentenceTransformer(model)
        self.prefix = "query: " if "e5" in model.lower() else ""

        self.norms = np.array([e.norm for e in self.entries])
        self.exact: dict[str, int] = {}
        for i, e in enumerate(self.entries):
            self.exact.setdefault(e.norm, i)

        self.emb = self._embeddings()
        self.words = self._build_word_index()

    # ---------------- indexes ----------------
    def _embeddings(self) -> np.ndarray:
        key = hashlib.sha256(
            (self.model_name + "\x00" + "\x00".join(e.norm for e in self.entries)).encode()
        ).hexdigest()
        cache = self.root / ".lis_emb_cache.npz"
        try:
            if cache.exists():
                z = np.load(cache, allow_pickle=False)
                if str(z["key"]) == key:
                    return z["emb"]
        except Exception:
            pass
        emb = self.encoder.encode([self.prefix + e.norm for e in self.entries],
                                  normalize_embeddings=True, show_progress_bar=False)
        try:
            np.savez(cache, key=key, emb=emb)
        except OSError:
            pass  # read-only dataset: no cache
        return emb

    def _build_word_index(self) -> dict[str, Path]:
        words: dict[str, Path] = {}
        for e in self.entries:
            is_word = e.kind == "word" or (e.kind is None and len(_tokens(e.norm)) == 1)
            if not is_word or e.kind == "sentence":
                continue
            lemma = (e.lemma or self.nlp(e.norm)[0].lemma_).lower()
            words.setdefault(lemma, e.path)      # same key: the first sample wins
            words.setdefault(e.norm, e.path)
        return words

    # ---------------- 1) semantic match ----------------
    @staticmethod
    def _guards_ok(a: str, b: str) -> bool:
        """Embeddings confuse 'vado' / 'non vado' and 'alle 19' / 'alle 20'."""
        return _has_neg(a) == _has_neg(b) and _numbers(a) == _numbers(b)

    def _probe(self, norm: str) -> Probe:
        if norm in self.exact:
            return Probe(self.exact[norm], 1.0, -1.0, True)
        q = self.encoder.encode([self.prefix + norm], normalize_embeddings=True,
                                show_progress_bar=False)[0]
        sims = self.emb @ q
        i = int(np.argmax(sims))
        other = self.norms != self.entries[i].norm   # ignore the other takes of the same sentence
        second = float(sims[other].max()) if other.any() else -1.0
        return Probe(i, float(sims[i]), second, self._guards_ok(norm, self.entries[i].norm))

    def _accepts(self, p: Probe, tau: float | None = None, margin: float | None = None) -> bool:
        tau = self.tau if tau is None else tau
        margin = self.margin if margin is None else margin
        return p.guards_ok and p.score >= tau and (p.score - p.second) >= margin

    # ---------------- 2) fallback: POS -> LIS order ----------------
    def _lis_order(self, raw: str) -> tuple[list[tuple[str, str]], list[str]]:
        doc = self.nlp(raw)
        ends_q = raw.strip().endswith("?")
        time_, subj, rest, obj, verb, neg, wh = ([] for _ in range(7))

        for t in doc:
            if t.is_punct or t.is_space:
                continue
            lemma, surf = t.lemma_.lower(), t.text.lower()
            item = (lemma, surf)
            if lemma in WH and (ends_q or t.i == 0):
                wh.append(item)
            elif t.pos_ in DROP_POS or lemma in DROP_LEMMAS:
                continue
            elif lemma in NEG or surf in NEG:
                neg.append(item)
            elif lemma in TIME or surf in TIME:
                time_.append(item)
            elif t.dep_.startswith("nsubj"):
                subj.append(item)
            elif t.pos_ == "VERB":
                verb.append(item)
            elif t.dep_ in {"obj", "iobj", "obl"}:
                obj.append(item)
            else:
                rest.append(item)

        mods = []
        if neg:
            mods.append("negation")
        if wh:
            mods.append("question_wh")
        elif ends_q:
            mods.append("question_yn")
        # time -> subject -> other -> object -> verb -> negation -> question word
        return time_ + subj + rest + obj + verb + neg + wh, mods

    # ---------------- API ----------------
    def translate(self, text: str) -> Result:
        norm = normalize(text)
        if not norm:
            return Result("none")

        p = self._probe(norm)
        entry = self.entries[p.idx]
        if self._accepts(p):
            return Result("sentence", [str(entry.path)], p.score, entry.sentence)

        order, mods = self._lis_order(text)
        paths, used, skipped = [], [], []
        for lemma, surf in order:
            hit = self.words.get(lemma) or self.words.get(surf)
            if hit is None:
                skipped.append(lemma)
            else:
                paths.append(str(hit))
                used.append(lemma)

        coverage = len(paths) / len(order) if order else 0.0
        if not paths or coverage < self.min_coverage:
            return Result("none", [], p.score, entry.sentence, [l for l, _ in order], skipped, mods)
        return Result("words", paths, p.score, entry.sentence, used, skipped, mods)

    # ---------------- threshold calibration ----------------
    def calibrate(self, cases: list[dict]) -> None:
        """cases: [{"text": "...", "expected": "dataset sentence" | null}, ...]
        expected=null means 'must not match any sentence'."""
        probes = []
        for c in cases:
            p = self._probe(normalize(c["text"]))
            exp = normalize(c["expected"]) if c.get("expected") else None
            probes.append((p, exp, self.entries[p.idx].norm))
        n_pos = sum(1 for _, e, _ in probes if e)
        print(f"{'tau':>5} {'accepted':>9} {'TP':>4} {'FP':>4} {'precision':>9} {'recall':>7}")
        for tau in np.arange(0.70, 1.0001, 0.01):
            tp = fp = acc = 0
            for p, exp, top in probes:
                if self._accepts(p, tau=float(tau)):
                    acc += 1
                    if exp and exp == top:
                        tp += 1
                    else:
                        fp += 1
            prec = tp / acc if acc else 1.0
            rec = tp / n_pos if n_pos else 0.0
            print(f"{tau:5.2f} {acc:9d} {tp:4d} {fp:4d} {prec:9.2f} {rec:7.2f}")
        print("\nPick the lowest tau with precision ~1.00: false positives (wrong animation) "
              "cost more than false negatives (they fall back to the word lookup).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("query")
    q.add_argument("dataset")
    q.add_argument("text")
    q.add_argument("--tau", type=float, default=0.88)
    q.add_argument("--margin", type=float, default=0.02)
    c = sub.add_parser("calibrate")
    c.add_argument("dataset")
    c.add_argument("cases", help="JSON: [{'text': ..., 'expected': ... | null}]")
    args = ap.parse_args()

    router = LisRouter(args.dataset, **({"tau": args.tau, "margin": args.margin}
                                        if args.cmd == "query" else {}))
    if args.cmd == "query":
        print(router.translate(args.text).to_json())
    else:
        router.calibrate(json.loads(Path(args.cases).read_text(encoding="utf-8")))
