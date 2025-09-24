import pathlib as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

##############################################################################
# 1) Pfad zu den CSV-Dateien  –  ANPASSEN!
##############################################################################
CSV_DIR = pl.Path(
    r"C:/Users/User/Desktop/Projekt_MedicalImaging_PC_V2/Prompt_Optimization_Dots_RQ2/Five_Runs/Complete_Dataset_V2"
)
csv_files = list(CSV_DIR.glob("*.csv"))
assert csv_files, f"⚠️  Kein CSV in {CSV_DIR}"

##############################################################################
# 2) CSVs einlesen + erste Spalte als Hash benennen
##############################################################################
df_all = pd.concat(
    [pd.read_csv(fp).assign(csv_name=fp.stem) for fp in csv_files],
    ignore_index=True,
)
df_all.columns = df_all.columns.str.strip()               # Whitespace entfernen
df_all.rename(columns={df_all.columns[0]: "hash"}, inplace=True)  # Hash-Spalte

##############################################################################
# 3) Accuracy pro Run (in %) berechnen  –  brauchen wir für nichts weiter,
#    aber falls du später Linienplots möchtest, bleibt es erhalten
##############################################################################
for i in range(1, 6):
    df_all[f"accuracy_run{i}"] = (
        df_all[f"correct_run{i}"]
        / (df_all[f"correct_run{i}"] + df_all[f"incorrect_run{i}"])
    ) * 100

##############################################################################
# 4) Welches Feld enthält RQ?  →  In der Beispieldatei ist das die Spalte "Model"
##############################################################################
RQ_COL = "Model"           # falls deine Spalte anders heißt: hier umbenennen!
assert RQ_COL in df_all.columns, f"Spalte '{RQ_COL}' nicht gefunden!"

##############################################################################
# 5) Für jede RQ separat 2 Plots erzeugen
##############################################################################
RQ_LIST = ["RQ1", "RQ2", "RQ3"]          # Reihenfolge, in der geplottet wird

for rq in RQ_LIST:
    df_rq = df_all[df_all[RQ_COL] == rq].copy()

    # ---------------- Mittelwert / Std je Prompt (Hash) bilden ----------------
    agg = (
        df_rq.groupby("hash", as_index=False)
        .agg(
            Accuracy_Mean=("Accuracy_Mean", "mean"),
            Accuracy_Std=("Accuracy_Std", "mean"),
            F1_Mean=("F1_Mean", "mean"),
            F1_Std=("F1_Std", "mean"),
        )
    )

    # Hash für X-Achse etwas kürzen (nur die ersten 6 Zeichen)
    agg["label"] = agg["hash"].map(name_map).fillna(agg["hash"].str[:6])

    # -------- Plot 1: Mean Accuracy ------------------------------------------
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))

    x = np.arange(len(agg))
    acc_mean = agg["Accuracy_Mean"] * 100
    acc_std  = agg["Accuracy_Std"]  * 100

    ax_acc.bar(x, acc_mean, yerr=acc_std, capsize=5)
    for i, val in enumerate(acc_mean):
        ax_acc.text(i, val + 0.5, f"{val:.2f}%", ha="center", va="bottom", fontsize=8)

    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(agg["label"], rotation=45, ha="right")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title(f"{rq}: Mean Accuracy (± Std) per Prompt")
    fig_acc.tight_layout()
    fig_acc.savefig(CSV_DIR / f"{rq.lower()}_accuracy_mean_std.png")
    plt.close(fig_acc)

    # -------- Plot 2: Mean F1 -------------------------------------------------
    fig_f1, ax_f1 = plt.subplots(figsize=(10, 6))

    f1_mean = agg["F1_Mean"] * 100
    f1_std  = agg["F1_Std"]  * 100

    ax_f1.bar(x, f1_mean, yerr=f1_std, capsize=5)
    for i, val in enumerate(f1_mean):
        ax_f1.text(i, val + 0.5, f"{val:.2f}%", ha="center", va="bottom", fontsize=8)

    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(agg["label"], rotation=45, ha="right")
    ax_f1.set_ylabel("F1-Score (%)")
    ax_f1.set_title(f"{rq}: Mean F1-Score (± Std) per Prompt")
    fig_f1.tight_layout()
    fig_f1.savefig(CSV_DIR / f"{rq.lower()}_f1_mean_std.png")
    plt.close(fig_f1)

##############################################################################
# 6) Zusätzlich: Plots pro Subexperiment für RQ2 und RQ3
##############################################################################

# for rq in ["RQ2", "RQ3"]:
#     df_rq = df_all[df_all[RQ_COL] == rq].copy()
#
#     sub_names = df_rq["Markers"].unique()
#     for sub in sub_names:
#         df_sub = df_rq[df_rq["Markers"] == sub].copy()
#
#         agg = (
#             df_sub.groupby("hash", as_index=False)
#             .agg(
#                 Accuracy_Mean=("Accuracy_Mean", "mean"),
#                 Accuracy_Std=("Accuracy_Std", "mean"),
#                 F1_Mean=("F1_Mean", "mean"),
#                 F1_Std=("F1_Std", "mean"),
#             )
#         )
#         agg["hash_short"] = agg["hash"].str[:6]
#         x = np.arange(len(agg))
#
#         # ------------------ Accuracy Plot ------------------
#         fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
#         acc_mean = agg["Accuracy_Mean"] * 100
#         acc_std = agg["Accuracy_Std"] * 100
#
#         ax_acc.bar(x, acc_mean, yerr=acc_std, capsize=5)
#         for i, val in enumerate(acc_mean):
#             ax_acc.text(i, val + 0.5, f"{val:.2f}%", ha="center", va="bottom", fontsize=8)
#
#         ax_acc.set_xticks(x)
#         ax_acc.set_xticklabels(agg["hash_short"], rotation=45, ha="right")
#         ax_acc.set_ylabel("Accuracy (%)")
#         ax_acc.set_title(f"{rq} - {sub}: Mean Accuracy (± Std)")
#         fig_acc.tight_layout()
#         fig_acc.savefig(CSV_DIR / f"{rq.lower()}_{sub}_accuracy.png")
#         plt.close(fig_acc)
#
#         # ------------------ F1 Score Plot ------------------
#         fig_f1, ax_f1 = plt.subplots(figsize=(10, 6))
#         f1_mean = agg["F1_Mean"] * 100
#         f1_std = agg["F1_Std"] * 100
#
#         ax_f1.bar(x, f1_mean, yerr=f1_std, capsize=5)
#         for i, val in enumerate(f1_mean):
#             ax_f1.text(i, val + 0.5, f"{val:.2f}%", ha="center", va="bottom", fontsize=8)
#
#         ax_f1.set_xticks(x)
#         ax_f1.set_xticklabels(agg["hash_short"], rotation=45, ha="right")
#         ax_f1.set_ylabel("F1-Score (%)")
#         ax_f1.set_title(f"{rq} - {sub}: Mean F1-Score (± Std)")
#         fig_f1.tight_layout()
#         fig_f1.savefig(CSV_DIR / f"{rq.lower()}_{sub}_f1.png")
#         plt.close(fig_f1)


print("✅ 18 Plots erfolgreich erstellt und gespeichert unter:", CSV_DIR)
