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

CSV_out_dir = pl.Path(r"C:/Users/User/Desktop/Projekt_MedicalImaging_PC_V2/Prompt_Optimization_Dots_RQ2/Five_Runs/Complete_Dataset_V2/Visualizations_Presentation")

##############################################################################
# 1) Pfad zu den CSV-Dateien – ANPASSEN!
##############################################################################
CSV_DIR = pl.Path(r"C:/Users/User/Desktop/Projekt_MedicalImaging_PC_V2/Prompt_Optimization_Dots_RQ2/Five_Runs/Complete_Dataset_V2")
csv_files = list(CSV_DIR.glob("*.csv"))
assert csv_files, f"Kein CSV in {CSV_DIR}"

##############################################################################
# 2) CSV einlesen, Accuracy je Run berechnen
##############################################################################
df_all = pd.concat(
    [pd.read_csv(fp).assign(prompt_name=fp.stem) for fp in csv_files],
    ignore_index=True
)

# Hash-ID benennen (erste Spalte)
df_all.rename(columns={df_all.columns[0]: "hash"}, inplace=True)

# Accuracy pro Run in %
for i in range(1, 6):
    df_all[f"accuracy_run{i}"] = (
        df_all[f"correct_run{i}"] /
        (df_all[f"correct_run{i}"] + df_all[f"incorrect_run{i}"])
    ) * 100

##############################################################################
# 3) Gruppieren pro Hash (Modell)
##############################################################################
agg_cols = {
    "Accuracy_Mean": "mean", "Accuracy_Std": "mean",
    "F1_Mean": "mean", "F1_Std": "mean",
}
for col in ["correct", "incorrect", "unsure"]:
    for i in range(1, 6):
        agg_cols[f"{col}_run{i}"] = "sum"
    agg_cols.update({f"{col}_run{i}": "sum" for i in range(1, 6)})

# Mittelwerte + Summe der Outcomes je Hash
df_plot = df_all.groupby("hash", as_index=False).agg(agg_cols)

# Accuracy pro Run (aus summierten Werten erneut berechnen)
for i in range(1, 6):
    df_plot[f"accuracy_run{i}"] = (
        df_plot[f"correct_run{i}"] /
        (df_plot[f"correct_run{i}"] + df_plot[f"incorrect_run{i}"])
    ) * 100

# Einmalige Label-Spalte für alle Plots
df_plot["label"] = df_plot["hash"].map(name_map).fillna(df_plot["hash"])

# (optional) Reihenfolge fixieren, damit alle Plots gleich sortiert sind
df_plot["label"] = pd.Categorical(df_plot["label"], categories=df_plot["label"], ordered=True)

##############################################################################
# 4) Linienplot – Accuracy-Verlauf über 5 Runs
##############################################################################
fig1, ax1 = plt.subplots()
runs = np.arange(1, 6)

for _, row in df_plot.iterrows():
    y = [row[f"accuracy_run{i}"] for i in runs]
    label = name_map.get(row["hash"], "Unbekannt")
    ax1.plot(runs, y, marker="o", label=row["label"])

ax1.set_xlabel("Run")
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Accuracy per Run and Prompt")
ax1.set_xticks(runs)
ax1.set_ylim(40, 85)
ax1.legend(fontsize="small", bbox_to_anchor=(1.02, 1))
fig1.tight_layout()
fig1.savefig(CSV_out_dir / "plot_accuracy_per_run.png")
plt.close(fig1)

##############################################################################
# 5) Balkenplot – Accuracy (Mean ± Std)
##############################################################################
fig2, ax2 = plt.subplots()
x = np.arange(len(df_plot))
acc_mean = df_plot["Accuracy_Mean"] * 100
acc_std = df_plot["Accuracy_Std"] * 100

ax2.bar(x, acc_mean, yerr=acc_std, capsize=5)
for i, val in enumerate(acc_mean):
    ax2.text(i, val + 0.5, f"{val:.2f}%", ha="center", va="bottom", fontsize=8)

ax2.set_xticks(x)
ax2.set_xticklabels(df_plot["label"], rotation=45, ha="right")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Accuracy (Mean ± Std) per Prompt")
fig2.tight_layout()
fig2.savefig(CSV_out_dir / "plot_accuracy_mean_std.png")
plt.close(fig2)

##############################################################################
# 6) Balkenplot – F1-Score (Mean ± Std)
##############################################################################
fig3, ax3 = plt.subplots()
f1_mean = df_plot["F1_Mean"] * 100
f1_std = df_plot["F1_Std"] * 100

ax3.bar(x, f1_mean, yerr=f1_std, capsize=5)
for i, val in enumerate(f1_mean):
    ax3.text(i, val + 0.5, f"{val:.2f}%", ha="center", va="bottom", fontsize=8)

ax3.set_xticks(x)
ax3.set_xticklabels(df_plot["label"], rotation=45, ha="right")
ax3.set_ylabel("F1-Score (%)")
ax3.set_title("F1-Score (Mean ± Std) per Prompt")
fig3.tight_layout()
fig3.savefig(CSV_out_dir / "plot_f1_mean_std.png")
plt.close(fig3)

##############################################################################
# 7) Stacked Bar Plot – Correct / Incorrect / Unsure (summiert über 5 Runs)
##############################################################################
fig4, ax4 = plt.subplots()
corr = df_plot[[f"correct_run{i}" for i in range(1, 6)]].sum(axis=1)
inc = df_plot[[f"incorrect_run{i}" for i in range(1, 6)]].sum(axis=1)
uns = df_plot[[f"unsure_run{i}" for i in range(1, 6)]].sum(axis=1)

ax4.bar(x, corr, label="Correct")
ax4.bar(x, inc, bottom=corr, label="Incorrect")
ax4.bar(x, uns, bottom=corr + inc, label="Unsure")

ax4.set_xticks(x)
ax4.set_xticklabels(df_plot["label"], rotation=45, ha="right")
ax4.set_ylabel("# Fälle (summiert über 5 Runs)")
ax4.set_title("Outcome-Verteilung je Modell")
ax4.legend()
fig4.tight_layout()
fig4.savefig(CSV_out_dir / "plot_outcomes_stacked.png")
plt.close(fig4)



print("✅ Alle Plots gespeichert unter:", CSV_out_dir)
