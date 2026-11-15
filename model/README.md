# LIS router: text → LIS animations

`lis_router.py` takes an Italian sentence and returns the dataset recordings to play, in order. The website joins them with smooth transitions and plays them on the avatar.

```
text ──► normalization ──► embedding ──► cosine vs dataset sentences
                                               │
                           score ≥ τ ──────────┴──► whole-sentence animation
                                               │
                           score < τ ──────────┴──► spaCy (POS + lemma)
                                                      ──► reorder into LIS order
                                                      ──► lemma lookup → word clip
                                                      ──► concatenation + blending (website)
```

## How it works

1. **Whole-sentence match.** A multilingual sentence encoder (`intfloat/multilingual-e5-base` by default) compares the input with every dataset sentence. The match is accepted when:
   - the cosine score is at least `tau` (default 0.88);
   - the gap to the best *different* sentence is at least `margin` (default 0.02);
   - negation and numbers agree (embeddings confuse "vado" / "non vado" and "alle 19" / "alle 20").

   Result: `mode="sentence"`, one sample folder.
2. **Word fallback.** spaCy (`it_core_news_sm` by default) gives POS tags and lemmas. Articles, auxiliaries, prepositions, conjunctions and the copula are dropped (LIS has no copula). The other words are reordered into LIS order: time → subject → other → object → verb → negation → question word. Each lemma is looked up among the single-word samples; missing words are skipped.

   Result: `mode="words"`, one folder per word, plus `modifiers` for the face (`negation`, `question_wh`, `question_yn`).
3. **Nothing found.** Result: `mode="none"`.

The sentence embeddings are cached in `<dataset>/.lis_emb_cache.npz` (skipped on a read-only dataset).

## Dataset layout

```
<root>/<any subfolder>/<sample_id>/metadata.json      ("sentence" field)
                                  /left_hand.csv, right_hand.csv, face.csv, body.csv
```

The returned path is the sample folder. Single-word clips are the samples whose `sentence` is one word. Optional overrides in `metadata.json`: `"type": "word" | "sentence"`, `"lemma": "<lemma>"`. The CSV files are the ones produced by [`dataset/`](../dataset/).

## Install

```bash
python installer.py                        # sentence-transformers, spaCy, numpy, spaCy model, encoder download
python installer.py --spacy-model it_core_news_md   # better lemmas
python installer.py --skip-model-download  # libraries and spaCy model only
```

The installer uses the current Python interpreter, so activate your virtual environment first. Python 3.9 or later is required.

## Usage

```bash
python lis_router.py query DATASET_ROOT "il treno parte alle 19"
python lis_router.py query DATASET_ROOT "ciao, come stai?" --tau 0.9 --margin 0.03
python lis_router.py calibrate DATASET_ROOT calibration.json
```

`query` prints the result as JSON (example):

```json
{
  "mode": "words",
  "paths": ["processed/output-chiara/0012_treno_...", "processed/output-chiara/0031_partire_..."],
  "score": 0.71,
  "matched_label": "il treno è in ritardo",
  "order": ["treno", "partire"],
  "skipped": ["19"],
  "modifiers": []
}
```

| Field | Meaning |
|---|---|
| `mode` | `sentence`, `words` or `none` |
| `paths` | sample folders to play, in order |
| `score` | best sentence similarity found |
| `matched_label` | best dataset sentence (also when rejected) |
| `order` | lemmas in LIS order (`words` mode) |
| `skipped` | lemmas without a clip |
| `modifiers` | face channels to overlay (negation, questions) |

`calibrate` reads `[{"text": "...", "expected": "dataset sentence" | null}, ...]` (`null` = must not match any sentence) and prints precision and recall for `tau` from 0.70 to 1.00. Pick the lowest `tau` with precision ≈ 1.00: a wrong animation costs more than a fallback to the word lookup.

## Use with the website

The website runs a TypeScript port of this router in the browser ([`website/src/router/lisRouter.ts`](../website/src/router/lisRouter.ts)), so it works as a static site (GitHub Pages) with no Python server. Its build exports `dataset/data/processed` for the browser (see [`website/README.md`](../website/README.md#the-dataset-in-the-site)). Differences from this script:

- no spaCy: function-word lists, irregular-verb table and the Snowball Italian stemmer stand in for POS tags and lemmas; subject and object come from the position before or after the verb; function words are signed when the dataset has a clip for them;
- the whole-sentence step uses exact, semantic (`multilingual-e5-small` in the browser, `e5-base` here, so `tau` may need a new calibration) and character similarity;
- English input is translated to Italian in the browser (`opus-mt-en-it`), and the English text also takes part in the semantic match;
- in the word fallback, multi-word dataset sentences found inside the text are used as one clip ("ciao, mi chiamo Chiara" → `ciao` + `mi chiamo`; this script plays only `ciao`).

This script stays the reference, and a server can still use it: the website accepts one clip per sign from `VITE_MOTION_MODEL_URL` (`clips` in the API, see [`website/README.md`](../website/README.md#api)).
