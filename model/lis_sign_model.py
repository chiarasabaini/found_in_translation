#!/usr/bin/env python3
"""
LIS (guanti MoCap + MediaPipe Face Mesh) -> frase italiana.

Layout dati atteso (label ricavata dal nome della cartella):
  processed/<firmatario>/<NNNN>_<frase_con_underscore>_<AAAAMMGG>_<HHMMSS>/
        left_hand.csv  right_hand.csv  face.csv  body.csv  [metadata.json]
  es. processed/output-chiara/0001_ciao_mi_chiamo_chiara_20261114_185050/
  Label: label.txt (se c'e') > campo "sentence" di metadata.json > slug del nome cartella.
  (stesso formato dei sample: colonne di servizio frame/timestamp/detected,
   guanti = 26 giunti x (pos xyz + quaternione xyzw), face = 478 landmark
   + 52 blendshape, body = collo/spalle/gomiti con visibilita')

Modi:
  inspect statistiche del dataset (clip per frase, firmatari, durate)
  dtw     baseline 1-NN con DTW (leave-one-out)
  cv      cross-validation stratificata del classificatore
  train   addestra su tutto e salva il checkpoint
  predict inferenza su una clip (top-k + rifiuto fuori vocabolario)
  t5      variante encoder from-scratch + mT5-small (LoRA), generativa

Esempi:
  python lis_sign_model.py --data processed --mode inspect
  python lis_sign_model.py --data processed --mode dtw
  python lis_sign_model.py --data processed --mode cv            # o --split signer
  python lis_sign_model.py --data processed --mode train --out model.pt
  python lis_sign_model.py --mode predict --ckpt model.pt --clip processed/output-chiara/0001_ciao_mi_chiamo_chiara_20261114_185050
"""
import argparse
import glob
import json
import math
import os
import re
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset

T = 96  # lunghezza fissa dopo il ricampionamento (cambiabile con --T)
STREAMS = ["left", "right", "face", "body"]

# sottoinsieme di landmark MediaPipe: labbra, sopracciglia, occhi, naso
FACE_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318,
            402, 317, 14, 87, 178, 88, 95, 78,
            70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
            33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, 1]


# ----------------------------------------------------------------- dati
META = ("frame", "timestamp_ms", "glove_timestamp_us", "hand_detected",
        "face_detected", "body_detected")


def quat_to_6d(q):
    """Quaternione (x,y,z,w) -> prime due colonne della matrice di rotazione.
    Continua e senza ambiguita' di segno (q e -q sono la stessa rotazione)."""
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    c1 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], -1)
    c2 = np.stack([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], -1)
    return np.concatenate([c1, c2], -1)


def load_hand(p):
    df = pd.read_csv(p)
    joints = sorted({c.rsplit("_pos_", 1)[0] for c in df.columns if "_pos_" in c})
    feats = []
    for j in joints:
        pos = df[[f"{j}_pos_{a}" for a in "xyz"]].to_numpy(np.float32)
        rot = df[[f"{j}_rot_{a}" for a in "xyzw"]].to_numpy(np.float32)
        feats += [pos, quat_to_6d(rot)]
    return np.nan_to_num(np.concatenate(feats, 1)).astype(np.float32)


def load_face(p):
    df = pd.read_csv(p)
    pts = np.stack([df[[f"face_{i}_{a}" for a in "xyz"]].to_numpy(np.float32)
                    for i in FACE_IDX], 1)  # (T, K, 3)
    nose = np.stack([df[[f"face_1_{a}" for a in "xyz"]].to_numpy(np.float32)], 1)
    eyes = np.stack([df[[f"face_33_{a}" for a in "xyz"]].to_numpy(np.float32),
                     df[[f"face_263_{a}" for a in "xyz"]].to_numpy(np.float32)], 1)
    scale = np.linalg.norm(eyes[:, 0] - eyes[:, 1], axis=-1).mean() + 1e-6
    geom = ((pts - nose) / scale).reshape(len(df), -1)
    bs = df[[c for c in df.columns if c.startswith("bs_")]].to_numpy(np.float32)
    return np.nan_to_num(np.concatenate([geom, bs], 1)).astype(np.float32)


def load_body(p):
    df = pd.read_csv(p)
    cols = [c for c in df.columns if c not in META and not c.startswith("neck_")
            or c == "neck_vis"]
    return np.nan_to_num(df[cols].to_numpy(np.float32))


def load_clip_dir(d):
    d = d.rstrip("/") + "/"
    return dict(left=load_hand(d + "left_hand.csv"), right=load_hand(d + "right_hand.csv"),
                face=load_face(d + "face.csv"), body=load_body(d + "body.csv"))


NAME_RE = re.compile(r"^(?P<idx>\d+)_(?P<slug>.+)_(?P<date>\d{8})_(?P<time>\d{6})$")


def load_clips(root):
    """Cerca ricorsivamente le cartelle con left_hand.csv; label dal nome cartella."""
    paths = sorted(os.path.dirname(p) for p in
                   glob.glob(os.path.join(root, "**", "left_hand.csv"), recursive=True))
    clips = []
    for d in paths:
        name = os.path.basename(d)
        m = NAME_RE.match(name)
        meta = {}
        mp = os.path.join(d, "metadata.json")
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                meta = json.load(f)
        if os.path.exists(os.path.join(d, "label.txt")):
            text = open(os.path.join(d, "label.txt"), encoding="utf-8").read().strip()
        elif meta.get("sentence"):
            text = meta["sentence"].strip()
        elif m:
            text = m.group("slug").replace("_", " ")
        else:
            print("salto (nome non riconosciuto):", d)
            continue
        rel = os.path.relpath(d, root).split(os.sep)
        signer = rel[0].replace("output-", "") if len(rel) > 1 else "?"
        c = load_clip_dir(d)
        c.update(text=text, path=d, signer=signer, frames=len(c["left"]),
                 fps=meta.get("avg_fps"), dur=meta.get("duration_ms"))
        if meta.get("num_frames") and meta["num_frames"] != c["frames"]:
            print(f"ATTENZIONE {name}: metadata dice {meta['num_frames']} frame, CSV ne ha {c['frames']}")
        clips.append(c)
    assert clips, f"nessuna clip trovata in {root}"
    vocab = sorted({c["text"] for c in clips})
    for c in clips:
        c["y"] = vocab.index(c["text"])
    return clips, vocab


def run_inspect(clips, vocab):
    from collections import Counter
    print(f"{len(clips)} clip | {len(vocab)} frasi | firmatari: {dict(Counter(c['signer'] for c in clips))}")
    cnt = Counter(c["text"] for c in clips)
    print("\nclip per frase (min/mediana/max): %d / %d / %d" % (
        min(cnt.values()), int(np.median(list(cnt.values()))), max(cnt.values())))
    for t, n in sorted(cnt.items(), key=lambda kv: kv[1]):
        print(f"  {n:3d}  {t}")
    fr = np.array([c["frames"] for c in clips])
    sec = np.array([(c["dur"] or c["frames"] / 30 * 1000) / 1000 for c in clips])
    fps = [c["fps"] for c in clips if c["fps"]]
    print(f"\ndurata clip: min {sec.min():.1f} s | mediana {np.median(sec):.1f} s | max {sec.max():.1f} s")
    print(f"fotogrammi: min {fr.min()} | mediana {int(np.median(fr))} | max {fr.max()}"
          + (f" | fps medio {np.mean(fps):.1f}" if fps else "")
          + f"  -> T attuale = {T} (clip mediana ricampionata x{T / np.median(fr):.2f})")
    if min(cnt.values()) < 3:
        print("ATTENZIONE: frasi con meno di 3 ripetizioni: la CV sara' molto rumorosa")


def resample(x, n):
    t = np.linspace(0, len(x) - 1, n)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    w = (t - i0)[:, None]
    return (x[i0] * (1 - w) + x[i1] * w).astype(np.float32)


def compute_stats(clips):
    stats = {}
    for k in STREAMS:
        a = np.concatenate([resample(c[k], T) for c in clips])
        stats[k] = (a.mean(0), np.maximum(a.std(0), 1e-2))
    return stats


class SignDS(Dataset):
    def __init__(self, clips, stats, train=False):
        self.clips, self.stats, self.train = clips, stats, train

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        c, out = self.clips[i], []
        for k in STREAMS:
            x = c[k]
            if self.train:
                n = len(x)
                a = int(n * random.uniform(0, 0.1))
                b = n - int(n * random.uniform(0, 0.1))
                x = x[a:max(b, a + 4)]
            x = resample(x, T)
            x = (x - self.stats[k][0]) / self.stats[k][1]
            if self.train:
                x = x * np.random.uniform(0.9, 1.1) + np.random.normal(0, 0.05, x.shape).astype(np.float32)
                x = x * (np.random.rand(1, x.shape[1]) > 0.1)  # channel dropout
                if random.random() < 0.5:  # time mask
                    s = random.randint(0, T - 8)
                    x[s:s + random.randint(2, 8)] = 0
                x = x.astype(np.float32)
            v = np.diff(x, axis=0, prepend=x[:1])
            out.append(torch.from_numpy(np.concatenate([x, v], 1).astype(np.float32)))
        return out, c["y"], c["text"]


def flat(c, stats):
    """Feature piatte per DTW: concatenazione dei flussi normalizzati."""
    return np.concatenate([(resample(c[k], T) - stats[k][0]) / stats[k][1] for k in STREAMS], 1)


# --------------------------------------------------------------- modelli
class SignEncoder(nn.Module):
    """3 flussi -> proiezione + conv locale -> self-attention sui token concatenati
    (la self-attention sul concatenato fa da cross-attention tra flussi)."""

    def __init__(self, dims, d=128, layers=3, heads=4, p=0.3):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(2 * f, d), nn.LayerNorm(d),
                                                 nn.GELU(), nn.Dropout(p)) for f in dims])
        self.conv = nn.ModuleList([nn.Conv1d(d, d, 5, padding=2) for _ in dims])
        self.pos = nn.Parameter(torch.randn(1, T, d) * 0.02)
        self.stype = nn.Parameter(torch.randn(len(dims), 1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, p, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, xs):
        toks = []
        for i, x in enumerate(xs):
            h = self.proj[i](x)
            h = h + self.conv[i](h.transpose(1, 2)).transpose(1, 2)
            toks.append(h + self.pos + self.stype[i])
        return self.enc(torch.cat(toks, 1))  # (B, 3T, d)


class SignClassifier(nn.Module):
    def __init__(self, dims, n_cls, d=128, **kw):
        super().__init__()
        self.enc = SignEncoder(dims, d, **kw)
        self.att = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, n_cls))

    def embed(self, xs):
        h = self.enc(xs)
        return (self.att(h).softmax(1) * h).sum(1)

    def forward(self, xs):
        return self.head(self.embed(xs))


class SignT5(nn.Module):
    """Encoder from-scratch -> bridge -> mT5-small (LoRA). Variante generativa."""

    def __init__(self, dims, name="google/mt5-small", d=128, lora=True):
        super().__init__()
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        self.tok = AutoTokenizer.from_pretrained(name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(name)
        self.enc = SignEncoder(dims, d)
        self.bridge = nn.Linear(d, self.t5.config.d_model)
        if lora:
            from peft import LoraConfig, get_peft_model
            self.t5 = get_peft_model(self.t5, LoraConfig(
                r=16, lora_alpha=32, target_modules=["q", "v"],
                lora_dropout=0.1, task_type="SEQ_2_SEQ_LM"))

    def forward(self, xs, texts):
        h = self.bridge(self.enc(xs))
        lab = self.tok(list(texts), return_tensors="pt", padding=True).input_ids.to(h.device)
        lab[lab == self.tok.pad_token_id] = -100
        return self.t5(inputs_embeds=h, labels=lab).loss

    @torch.no_grad()
    def generate(self, xs, beams=4):
        h = self.bridge(self.enc(xs))
        out = self.t5.generate(inputs_embeds=h, max_new_tokens=48, num_beams=beams)
        return self.tok.batch_decode(out, skip_special_tokens=True)


# ---------------------------------------------------------------- training
def dims_of(clips):
    return [clips[0][k].shape[1] for k in STREAMS]


def to_dev(xs, dev):
    return [x.to(dev) for x in xs]


def make_sched(opt, total):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))


def fit(train_clips, n_cls, a, dev):
    stats = compute_stats(train_clips)
    dl = DataLoader(SignDS(train_clips, stats, True), batch_size=a.bs,
                    shuffle=True, drop_last=len(train_clips) > a.bs)
    model = SignClassifier(dims_of(train_clips), n_cls, a.d).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = make_sched(opt, a.epochs * len(dl))
    for ep in range(a.epochs):
        model.train()
        for xs, y, _ in dl:
            loss = F.cross_entropy(model(to_dev(xs, dev)), y.to(dev), label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    return model, stats


@torch.no_grad()
def evaluate(model, stats, clips, dev, ks=(1, 3)):
    model.eval()
    hits = {k: 0 for k in ks}
    for xs, y, _ in DataLoader(SignDS(clips, stats), batch_size=16):
        top = model(to_dev(xs, dev)).topk(max(ks), 1).indices.cpu()
        for k in ks:
            hits[k] += (top[:, :k] == y[:, None]).any(1).sum().item()
    return {k: v / len(clips) for k, v in hits.items()}


def folds(clips, k, split="stratified"):
    ys = [c["y"] for c in clips]
    if split == "signer":  # valuta su firmatari mai visti
        groups = [c["signer"] for c in clips]
        return GroupKFold(min(k, len(set(groups)))).split(clips, ys, groups)
    cnt = np.bincount(ys)
    m = min(cnt[cnt > 0])
    if m >= 2:
        return StratifiedKFold(min(k, m), shuffle=True, random_state=0).split(clips, ys)
    return KFold(k, shuffle=True, random_state=0).split(clips)


def run_cv(clips, vocab, a, dev):
    res = []
    for i, (tr, te) in enumerate(folds(clips, a.folds, a.split)):
        tr_c, te_c = [clips[j] for j in tr], [clips[j] for j in te]
        model, stats = fit(tr_c, len(vocab), a, dev)
        r = evaluate(model, stats, te_c, dev)
        res.append(r)
        print(f"fold {i}: top1={r[1]:.3f} top3={r[3]:.3f}")
    print("media top1=%.3f top3=%.3f" % (np.mean([r[1] for r in res]), np.mean([r[3] for r in res])))


def run_dtw(clips):
    stats = compute_stats(clips)
    X = [flat(c, stats) for c in clips]
    ok = 0
    for i in range(len(clips)):
        best, bj = np.inf, -1
        for j in range(len(clips)):
            if j != i:
                d = dtw(X[i], X[j])
                if d < best:
                    best, bj = d, j
        ok += clips[bj]["y"] == clips[i]["y"]
    print(f"DTW 1-NN leave-one-out acc = {ok / len(clips):.3f}")


def dtw(a, b, band=16):
    n, m = len(a), len(b)
    C = cdist(a, b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m] / (n + m)


def run_t5(clips, a, dev):
    tr, te = next(folds(clips, a.folds, a.split))
    tr_c, te_c = [clips[j] for j in tr], [clips[j] for j in te]
    stats = compute_stats(tr_c)
    model = SignT5(dims_of(clips)).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-2)
    dl = DataLoader(SignDS(tr_c, stats, True), batch_size=a.bs, shuffle=True)
    sched = make_sched(opt, a.epochs * len(dl))
    for ep in range(a.epochs):
        model.train()
        for xs, _, texts in dl:
            loss = model(to_dev(xs, dev), texts)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
        if ep % 10 == 0:
            print(f"ep {ep} loss {loss.item():.3f}")
    model.eval()
    em = 0
    for xs, _, texts in DataLoader(SignDS(te_c, stats), batch_size=8):
        preds = model.generate(to_dev(xs, dev))
        for p, t in zip(preds, texts):
            em += p.strip().lower() == t.strip().lower()
            print(f"GT: {t}\nPR: {p}\n")
    print(f"exact match = {em / len(te_c):.3f}")


# ---------------------------------------------------------------- inference
def save_ckpt(model, stats, vocab, dims, a, path):
    torch.save(dict(sd=model.state_dict(), stats=stats, vocab=vocab, dims=dims, d=a.d, T=T), path)


@torch.no_grad()
def predict(ckpt_path, clip_dir, topk=3, reject_below=0.35, dev="cpu"):
    global T
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    T = ck.get("T", T)
    model = SignClassifier(ck["dims"], len(ck["vocab"]), ck["d"]).to(dev)
    model.load_state_dict(ck["sd"])
    model.eval()
    clip = dict(load_clip_dir(clip_dir), y=0, text="")
    xs, _, _ = SignDS([clip], ck["stats"])[0]
    probs = model([x[None].to(dev) for x in xs]).softmax(-1)[0]
    p, idx = probs.topk(topk)
    res = [(ck["vocab"][i], float(q)) for q, i in zip(p, idx)]
    if res[0][1] < reject_below:
        return "[frase non riconosciuta]", res
    return res[0][0], res


# --------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--mode", choices=["inspect", "dtw", "cv", "train", "predict", "t5"], default="cv")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--T", type=int, default=T, help="passi temporali dopo il ricampionamento")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split", choices=["stratified", "signer"], default="stratified",
                    help="cv: stratificato per frase, oppure leave-signer-out")
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--ckpt", default="model.pt")
    ap.add_argument("--clip")
    a = ap.parse_args()

    T = a.T
    random.seed(0), np.random.seed(0), torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if a.mode == "predict":
        text, top = predict(a.ckpt, a.clip, dev=dev)
        print(text, top)
    else:
        clips, vocab = load_clips(a.data)
        print(f"{len(clips)} clip, {len(vocab)} frasi, device={dev}")
        if a.mode == "inspect":
            run_inspect(clips, vocab)
        elif a.mode == "dtw":
            run_dtw(clips)
        elif a.mode == "cv":
            run_cv(clips, vocab, a, dev)
        elif a.mode == "t5":
            run_t5(clips, a, dev)
        else:
            model, stats = fit(clips, len(vocab), a, dev)
            save_ckpt(model, stats, vocab, dims_of(clips), a, a.out)
            print("salvato", a.out)
