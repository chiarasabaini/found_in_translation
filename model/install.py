#!/usr/bin/env python3
"""
Installer per lis_sign_model.py (Linux / Windows / macOS).

  python install.py                 # crea .venv, installa torch adatto + requirements
  python install.py --torch cpu     # forza torch solo CPU
  python install.py --torch skip    # non installa torch (lo gestisci tu)
  python install.py --torch-index https://download.pytorch.org/whl/cu126   # indice personalizzato
  python install.py --venv myenv

Scelta di torch con --torch auto (default):
  - nessuna GPU NVIDIA  -> build CPU (leggera)
  - Linux + NVIDIA      -> pacchetto standard di PyPI (include CUDA)
  - --torch skip: peft installa comunque torch (da PyPI) se non gia' presente
  - Windows + NVIDIA    -> indice CUDA di PyTorch (cu126); se non combacia col tuo
                           driver, rilancia con --torch-index <url> (vedi pytorch.org)
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import venv

ROOT = os.path.dirname(os.path.abspath(__file__))
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
WIN_CUDA_INDEX = "https://download.pytorch.org/whl/cu126"


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def has_nvidia():
    return shutil.which("nvidia-smi") is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", default=".venv")
    ap.add_argument("--torch", choices=["auto", "cpu", "skip"], default="auto")
    ap.add_argument("--torch-index", help="URL indice pip per torch (ha la precedenza)")
    a = ap.parse_args()

    if sys.version_info < (3, 9):
        sys.exit("Serve Python >= 3.9")

    env_dir = os.path.join(ROOT, a.venv) if not os.path.isabs(a.venv) else a.venv
    if not os.path.exists(env_dir):
        print(f"Creo l'ambiente virtuale in {env_dir}")
        venv.EnvBuilder(with_pip=True).create(env_dir)
    win = platform.system() == "Windows"
    py = os.path.join(env_dir, "Scripts" if win else "bin", "python.exe" if win else "python")

    run([py, "-m", "pip", "install", "--upgrade", "pip"])

    # 1) torch per primo, con l'indice giusto
    if a.torch != "skip":
        index = a.torch_index
        if not index:
            if a.torch == "cpu" or not has_nvidia():
                index = CPU_INDEX
            elif win:
                index = WIN_CUDA_INDEX
        cmd = [py, "-m", "pip", "install", "torch"]
        if index:
            cmd += ["--index-url", index]
        run(cmd)

    # 2) il resto (torch escluso: e' gia' gestito sopra, o saltato con --torch skip)
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as f:
        lines = [l for l in f if not l.strip().lower().startswith("torch")]
    tmp_req = os.path.join(env_dir, "requirements_no_torch.txt")
    with open(tmp_req, "w", encoding="utf-8") as f:
        f.writelines(lines)
    run([py, "-m", "pip", "install", "-r", tmp_req])

    # 3) verifica
    if a.torch == "skip":
        check = "import numpy, pandas, scipy, sklearn, transformers, peft; print('dipendenze OK (torch saltato)')"
    else:
        check = (
            "import torch, numpy, pandas, scipy, sklearn, transformers, peft, sys;"
            "print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available());"
            "sys.path.insert(0, r'%s');"
            "import lis_sign_model; print('lis_sign_model importato correttamente')" % ROOT
        )
    run([py, "-c", check])

    act = os.path.join(a.venv, "Scripts", "activate") if win else f"source {a.venv}/bin/activate"
    print("\nFatto. Attiva l'ambiente con:\n  " + act)
    print("Poi, ad esempio:\n  python lis_sign_model.py --data data --mode dtw")


if __name__ == "__main__":
    main()
