"""
installer.py - installa tutto il necessario per lis_router.py.

Esegue, con l'interprete Python corrente (quindi dentro il venv attivo):
  1. pip install sentence-transformers spacy numpy
  2. python -m spacy download <modello spaCy>
  3. scarica in anticipo il modello del sentence encoder (cache di Hugging Face),
     cosi' la prima query non resta in attesa del download

Uso:
  python installer.py
  python installer.py --spacy-model it_core_news_md
  python installer.py --st-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
  python installer.py --skip-model-download      # solo librerie + spaCy
"""
from __future__ import annotations

import argparse
import subprocess
import sys

PACKAGES = ["numpy", "sentence-transformers", "spacy"]


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--spacy-model", default="it_core_news_sm",
                    help="it_core_news_sm | _md | _lg (default: it_core_news_sm)")
    ap.add_argument("--st-model", default="intfloat/multilingual-e5-base",
                    help="modello sentence-transformers (default: intfloat/multilingual-e5-base)")
    ap.add_argument("--skip-model-download", action="store_true",
                    help="non pre-scaricare il modello del sentence encoder")
    args = ap.parse_args()

    if sys.version_info < (3, 9):
        print("Serve Python 3.9 o superiore.", file=sys.stderr)
        return 1

    try:
        run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        run([sys.executable, "-m", "pip", "install", "--upgrade", *PACKAGES])
        run([sys.executable, "-m", "spacy", "download", args.spacy_model])

        if not args.skip_model_download:
            print(f"\nScarico il modello {args.st_model} ...", flush=True)
            from sentence_transformers import SentenceTransformer
            SentenceTransformer(args.st_model)
    except subprocess.CalledProcessError as e:
        print(f"\nComando fallito (codice {e.returncode}): {' '.join(map(str, e.cmd))}", file=sys.stderr)
        return e.returncode or 1
    except Exception as e:  # download del modello, rete, ecc.
        print(f"\nErrore: {e}", file=sys.stderr)
        return 1

    print("\nInstallazione completata. Prova con:")
    print('  python lis_router.py query DATASET_ROOT "il treno parte alle 19"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
