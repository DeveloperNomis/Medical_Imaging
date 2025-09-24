#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Erzeugt pro Prompt-Ordner und pro Research-Question (RQ1, RQ2, RQ3)
eine Confusion-Matrix (2×2) als PNG.

Aufruf:
    python make_cm_per_rq.py <base_dir>  <out_dir>

    <base_dir>  … Hash-Ordner, darin RQ1/RQ2/RQ3-Unterordner
    <out_dir>   … Zielverzeichnis für die PNG-Dateien
"""

from __future__ import annotations            # erlaubt int|None ab Py 3.7
import argparse, json, re
from pathlib import Path
from typing import Optional, List, Tuple
from matplotlib.colors import ListedColormap

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

name_map = {
    "3b85c67999b282be7ff2c4ccdafd51b1": "No Role + Original Prompt",
    "f6cf7d9dfe641e861e1c6d95286581be": "No Role + Anatomy",
    "ccbb6cdf99987e8e6eb35a5bda5020b5": "No Role + New spatial",
    "52060f10fbaf3952d02e5ce518f0857a": "Experienced Radiologist + Original Prompt",
    "73fdfe2f6eca668b542f0b7fc88f8d8f": "Experienced Radiologist + Anatomy",
    "022c477dd5ec0d142a68314623390ba8": "Experienced Radiologist + New spatial",
    "5ec347eef3693ca171e0287d456a708b": "VR Verifier + Original prompt",
    "8f077c27fc7d3292dc34dcfd5cde0712": "VR Verifier + Anatomy",
    "3bb674764c297465f5800633f51f136a": "VR Verifier + New spatial"
}

# ---------------------------------------------------------------------------
# 1) Kleine Helfer
# ---------------------------------------------------------------------------
_DIG_RE  = re.compile(r'^\s*(0|1)\s*$')
_YESNO_RE = re.compile(r'^\s*(yes|no)\s*$', re.I)
_END_RE  = re.compile(r'(0|1)[\s\.\)\]\}]*$', re.I)

def parse_answer(text: str) -> Optional[int]:
    """Versucht, ein Modell-Antwort-Snippet in 0/1 zu überführen."""
    if m := _DIG_RE.match(text):
        return int(m.group(1))

    if m := _YESNO_RE.match(text):
        return 1 if m.group(1).lower() == "yes" else 0

    if m := _END_RE.search(text):
        return int(m.group(1))

    return None


def collect_preds_targets(json_file: Path) -> Tuple[List[int], List[int]]:
    """Liest ein JSON (Format wie in der Frage) und sammelt y_pred / y_true."""
    preds, trues = [], []
    data = json.loads(json_file.read_text(encoding="utf-8"))

    for img in data:
        for res in img["results_call"]:
            exp = res["expected_answer"]
            pa  = parse_answer(res["model_answer"])
            if pa is None:                              # ungeparst → als falsch werten
                pa = 0 if exp == 1 else 1
            preds.append(pa)
            trues.append(exp)
    return preds, trues


def plot_cm(cm: np.ndarray,
            title: str,
            precision: float,
            recall: float,
            f1: float,
            out_path: Path) -> None:
    """Speichert eine hübsche 2×2-Matrix als PNG."""
    fig, ax = plt.subplots(figsize=(6, 6))

    DIAG_COLOR = "#5fa8ff"  # EIN einheitliches Hell-Blau
    WHITE = "#ffffff"

    vmax = cm.max() if cm.max() > 0 else 1  # Skala ≥1 für imshow
    diag_mask = np.eye(2, dtype=bool)

    # Diagonale (TP + TN) – einfarbig hell-blau
    diag_img = np.where(np.eye(2, dtype=bool), 1, np.nan)
    ax.imshow(diag_img,
              cmap=ListedColormap([WHITE, DIAG_COLOR]),  # 0 → weiß, 1 → hell-blau
              vmin=0, vmax=1)

    # ---------------- Off-Diagonale (weiß) -------------
    off_only = cm.astype(float).copy()
    off_only[diag_mask] = np.nan  # Diagonale ausblenden
    ax.imshow(off_only,
              cmap=ListedColormap(["white"]),
              vmin=0, vmax=vmax)

    # Werte eintragen
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha='center', va='center',
                fontsize=18, fontweight='bold',
                color='black')

    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    n = cm.sum()
    ax.set_title(f"{title}\nP={precision:.2f}  R={recall:.2f}  F1={f1:.2f}  N={n}",
                 fontsize=14, pad=20)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def safe_filename(s: str) -> str:
    # Windows-sichere Dateinamen
    s = re.sub(r'[<>:"/\\|?*]+', "_", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# 2) Hauptlogik
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Confusion-Matrizen pro RQ erzeugen")
    parser.add_argument("base_dir", type=Path, help="Ordner mit Hash-Prompts")
    parser.add_argument("out_dir",  type=Path, help="Zielordner für PNGs")
    args = parser.parse_args()

    base_dir: Path = args.base_dir.expanduser().resolve()
    out_dir:  Path = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rq_list = ["RQ1", "RQ2", "RQ3"]

    # Jeder Prompt-Ordner (Hash)
    for prompt_dir in sorted([d for d in base_dir.iterdir() if d.is_dir()]):
        prompt_hash = prompt_dir.name
        label = name_map.get(prompt_hash, prompt_hash)  # Fallback: Hash
        stub = safe_filename(f"{label}__{prompt_hash[:6]}")  # kollisionssicherer Dateistub

        for rq in rq_list:
            rq_dir = prompt_dir / rq
            if not rq_dir.is_dir():
                continue  # RQ-Ordner existiert nicht

            # Alle JSONs der Subexperimente & Runs
            json_files = list(rq_dir.rglob("*.json"))
            if not json_files:
                continue

            preds, trues = [], []
            for jf in json_files:
                p, t = collect_preds_targets(jf)
                preds.extend(p)
                trues.extend(t)

            if not preds:
                continue  # nichts gefunden

            y_true = np.array(trues)
            y_pred = np.array(preds)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            prec = precision_score(y_true, y_pred, zero_division=0)
            rec  = recall_score(y_true, y_pred, zero_division=0)
            f1   = f1_score(y_true, y_pred, zero_division=0)

            out_file = out_dir / f"{stub}_{rq}_cm.png"
            plot_cm(cm, f"{label} – {rq}", prec, rec, f1, out_file)
            print(f"✓  Confusion-Matrix gespeichert: {out_file.relative_to(out_dir.parent)}")


if __name__ == "__main__":
    main()
