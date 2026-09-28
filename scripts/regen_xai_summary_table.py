"""Regenerate panel E — XAI Metrics Summary — as a standalone figure.

Usage:
    .venv/bin/python regen_xai_summary_table.py
"""

import pandas as pd
import matplotlib.pyplot as plt

from src.gtzan_probe.config import DATA_DIR, setup_plotting, savefig

setup_plotting()

summary_df = pd.read_csv(DATA_DIR / "xai_summary_table.csv")
print("Loaded xai_summary_table.csv")

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")

table = ax.table(
    cellText=summary_df.round(3).values,
    colLabels=summary_df.columns,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.6)

ax.set_title("E  XAI Metrics Summary", fontweight="bold", pad=12)

savefig("FINAL_E_xai_metrics_summary", fig)
print("Done.")
