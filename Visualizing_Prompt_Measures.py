##############################################################################
# Prompt-Vergleich – 5-Run-Plots
# ------------------------------------------------------------
# • Linienplot  : Accuracy pro Run
# • Balkenplot  : Accuracy (Mean ± Std)
# • Balkenplot  : F1-Score (Mean ± Std)
# • StackedBars : Correct / Incorrect / Unsure (5 Runs aufsummiert)
##############################################################################

import pathlib as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# 1) Pfad zu den CSV-Dateien – ANPASSEN!
##############################################################################
CSV_DIR = pl.Path(r"C:/Users/User/Desktop/Projekt_MedicalImaging_PC/Prompt_Optimization_Dots_RQ2/Five_Runs/Complete_Dataset")
csv_files = list(CSV_DIR.glob("*.csv"))
assert csv_files, f"Kein CSV in {CSV_DIR}"

##############################################################################
# 2) Alle Dateien laden, Accuracy je Run hinzufügen
##############################################################################
df_all = pd.concat(
    [pd.read_csv(fp).assign(prompt_name=fp.stem) for fp in csv_files],
    ignore_index=True
)

# Accuracy pro Run in %
for i in range(1, 6):
    df_all[f"accuracy_run{i}"] = (
        df_all[f"correct_run{i}"] /
        (df_all[f"correct_run{i}"] + df_all[f"incorrect_run{i}"])
    ) * 100

##############################################################################
# 3) Eine Zeile pro Prompt – Mittel über evtl. Duplikate bilden
##############################################################################
agg_cols_mean = [
    "Accuracy_Mean", "F1_Mean",
    "Accuracy_LeftRight_Mean", "F1_LeftRight_Mean",
    "Accuracy_AboveBelow_Mean", "F1_AboveBelow_Mean"
]
agg_cols_std = [
    "Accuracy_Std", "F1_Std",
    "Accuracy_LeftRight_Std", "F1_LeftRight_Std",
    "Accuracy_AboveBelow_Std", "F1_AboveBelow_Std"
]

df_plot = (
    df_all
    .groupby("prompt_name", as_index=False)
    .agg({**{c: "mean" for c in agg_cols_mean},
          **{c: "mean" for c in agg_cols_std}})
)

##############################################################################
# 4) Linienplot – Accuracy-Verlauf über 5 Runs
##############################################################################
fig1, ax1 = plt.subplots()

runs = np.arange(1, 6)
for _, row in df_all.groupby("prompt_name").first().iterrows():
    y = [row[f"accuracy_run{i}"] for i in runs]
    ax1.plot(runs, y, marker="o", label=row.name)

ax1.set_xlabel("Run")
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Accuracy pro Run und Prompt")
ax1.set_xticks(runs)
ax1.set_ylim(40, 70)
ax1.legend(fontsize="small", bbox_to_anchor=(1.02, 1))
fig1.tight_layout()
fig1.savefig(CSV_DIR / "plot_accuracy_per_run.png")
plt.close(fig1)

##############################################################################
# 5) Balkenplot – Accuracy (Mean ± Std)
##############################################################################
fig2, ax2 = plt.subplots()

x = np.arange(len(df_plot))
acc_mean = df_plot["Accuracy_Mean"] * 100
acc_std  = df_plot["Accuracy_Std"]  * 100

ax2.bar(x, acc_mean, yerr=acc_std, capsize=5)

for i, val in enumerate(acc_mean):
    ax2.text(i, val + 0.1, f"{val:.2f}%", ha="center", va="bottom", fontsize=8)

ax2.set_xticks(x)
ax2.set_xticklabels(df_plot["prompt_name"], rotation=45, ha="right")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Accuracy (Mean ± Std) je Prompt")
fig2.tight_layout()
fig2.savefig(CSV_DIR / "plot_accuracy_mean_std.png")
plt.close(fig2)

##############################################################################
# 6) Balkenplot – F1 (Mean ± Std)
##############################################################################
fig3, ax3 = plt.subplots()

f1_mean = df_plot["F1_Mean"] * 100
f1_std  = df_plot["F1_Std"]  * 100

ax3.bar(x, f1_mean, yerr=f1_std, capsize=5)

for i, val in enumerate(f1_mean):
    ax3.text(i, val + 0.1, f"{val:.2f}%", ha="center", va="bottom", fontsize=8)

ax3.set_xticks(x)
ax3.set_xticklabels(df_plot["prompt_name"], rotation=45, ha="right")
ax3.set_ylabel("F1-Score (%)")
ax3.set_title("F1-Score (Mean ± Std) je Prompt")
fig3.tight_layout()
fig3.savefig(CSV_DIR / "plot_f1_mean_std.png")
plt.close(fig3)

##############################################################################
# 7) Stacked-Bars – Correct / Incorrect / Unsure (5 Runs summiert)
##############################################################################
fig4, ax4 = plt.subplots()

corr = df_all.groupby("prompt_name")[[f"correct_run{i}"   for i in range(1, 6)]].sum().sum(axis=1)
inc  = df_all.groupby("prompt_name")[[f"incorrect_run{i}" for i in range(1, 6)]].sum().sum(axis=1)
uns  = df_all.groupby("prompt_name")[[f"unsure_run{i}"    for i in range(1, 6)]].sum().sum(axis=1)

x = np.arange(len(corr))
ax4.bar(x, corr,                 label="Correct")
ax4.bar(x, inc,  bottom=corr,    label="Incorrect")
ax4.bar(x, uns,  bottom=corr+inc, label="Unsure")

ax4.set_xticks(x)
ax4.set_xticklabels(corr.index, rotation=45, ha="right")
ax4.set_ylabel("# Fälle (5 Runs)")
ax4.set_title("Outcome-Verteilung pro Prompt")
ax4.legend()
fig4.tight_layout()
fig4.savefig(CSV_DIR / "plot_outcomes_stacked.png")
plt.close(fig4)

print("✅ Alle Plots gespeichert unter:", CSV_DIR)
