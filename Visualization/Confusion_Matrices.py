"""
Erzeugt pro Prompt-Ordner (Hash) genau eine Confusion-Matrix als PNG.
Aufrufbeispiel:
    python make_conf_mats.py "C:\Pfad\zur\Complete_Dataset_V2" -o "C:\…\ConfMats"
"""
import json, re, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Optional
from matplotlib.colors import ListedColormap

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

# ---------- Antwort-Parser --------------------------------------------------
DIGIT_RE  = re.compile(r'^\s*(0|1)\s*$')
YESNO_RE  = re.compile(r'^\s*(yes|no)\s*$', re.I)
END_RE    = re.compile(r'(0|1)[\s\.\)\]\}]*$')

def parse_answer(ans: str) -> Optional[int]:
    """
    Extrahiert 0/1 aus Modell­antworten wie
      "A: 1", "Answer: 0", "yes", "1", …
    Gibt None zurück, wenn nichts Passendes gefunden wurde.
    """
    # Label am Anfang entfernen ("A:", "Answer:", …)
    ans = re.sub(r'^\s*[Aa]\w*\s*[:\-]\s*', '', ans.strip())

    if m := DIGIT_RE.match(ans):
        return int(m.group(1))
    if m := YESNO_RE.match(ans):
        return 1 if m.group(1).lower() == "yes" else 0
    if m := END_RE.search(ans):
        return int(m.group(1))
    return None


# ---------- Plot-Funktion ---------------------------------------------------
def plot_confmat(cm: np.ndarray, title: str, outfile: Path) -> None:
    tn, fp, fn, tp = cm.ravel()
    P  = tp / (tp + fp) if (tp + fp) else 0
    R  = tp / (tp + fn) if (tp + fn) else 0
    F1 = 2*P*R / (P+R)  if (P+R)     else 0

    DIAG_COLOR = "#5fa8ff"  # EIN einheitliches Hell-Blau
    WHITE = "#ffffff"

    vmax = cm.max() if cm.max() else 1  # für Farbnormierung
    diag_mask = np.eye(2, dtype=bool)

    fig, ax = plt.subplots(figsize=(6,6))
    # Diagonale (TP + TN) – einfarbig hell-blau
    diag_img = np.where(np.eye(2, dtype=bool), 1, np.nan)
    ax.imshow(diag_img,
              cmap=ListedColormap([WHITE, DIAG_COLOR]),  # 0 → weiß, 1 → hell-blau
              vmin=0, vmax=1)

    # Off-Diagonale weiß
    off_only = cm.astype(float).copy()
    off_only[diag_mask] = np.nan
    ax.imshow(off_only, cmap=ListedColormap(["white"]), vmin=0, vmax=vmax)



    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(f"{title}\nP={P:.2f}  R={R:.2f}  F1={F1:.2f}  N={cm.sum():,}")

    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="black", fontweight="bold", fontsize=14)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


# ---------- Datensammlung ---------------------------------------------------
def collect_preds_targets(prompt_dir: Path):
    preds, targs = [], []
    for jf in prompt_dir.rglob("*.json"):
        if jf.name.startswith("Result_"):          # Zusammenfassungen überspringen
            continue
        data = json.loads(jf.read_text(encoding="utf-8"))
        for entry in data:
            for res in entry["results_call"]:
                targ = res["expected_answer"]
                pred = parse_answer(res["model_answer"])
                if pred is None:                   # ungeparst ⇒ falsch werten
                    pred = 1 - targ
                preds.append(pred); targs.append(targ)
    return preds, targs


def safe_filename(s: str) -> str:
    # Windows-sichere Dateinamen
    s = re.sub(r'[<>:"/\\|?*]+', "_", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------- MAIN ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir", type=Path,
                    help="Wurzelverzeichnis mit Hash-Unterordnern (Prompt-Ordnern)")
    ap.add_argument("-o", "--out", type=Path, default="Confusion_Matrices",
                    help="Zielordner für PNG-Dateien")
    args = ap.parse_args()

    out_root: Path = args.out
    any_done = False

    for prompt_dir in sorted(p for p in args.base_dir.iterdir() if p.is_dir()):
        preds, targs = collect_preds_targets(prompt_dir)
        if not preds:
            print(f"[{prompt_dir.name}] – keine JSON-Files gefunden, übersprungen.")
            continue
        cm = confusion_matrix(targs, preds, labels=[0,1])
        hash_id = prompt_dir.name
        label = name_map.get(hash_id, hash_id)  # << hier mappen
        title = f"{label}"  # (optional: f"{label} ({hash_id[:6]})")
        fname = safe_filename(f"{label}.png")  # Kollision vermeiden?

        plot_confmat(cm, title, args.out / fname)
        print(f"[{hash_id}]  {fname} gespeichert.")
        any_done = True

    if any_done:
        print(f"\n✓ Alle Matrizen unter: {out_root.resolve()}")
    else:
        print("⚠️  Keine gültigen Daten gefunden.")


if __name__ == "__main__":
    main()