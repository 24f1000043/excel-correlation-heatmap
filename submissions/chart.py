# chart.py
# Author: 24f1000043@ds.study.iitm.ac.in
# Generates a publication-ready Seaborn correlation heatmap and saves as 512x512 PNG.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------
# 1. Generate realistic synthetic data
# -------------------------
# Business context: customer engagement metrics for many customers
rng = np.random.default_rng(42)
n_customers = 500

data = pd.DataFrame({
    "Visits": rng.poisson(lam=8, size=n_customers) + rng.normal(0, 1, n_customers),
    "Avg_Spend": rng.normal(50, 20, n_customers).clip(1),
    "Session_Duration": rng.normal(300, 80, n_customers).clip(10),  # seconds
    "Pages_Per_Session": rng.normal(4, 1.2, n_customers).clip(1),
    "Repeat_Rate": rng.beta(2, 5, n_customers),   # 0-1 fraction
    "NPS": rng.normal(30, 15, n_customers),      # Net Promoter Score
    "Support_Calls": rng.poisson(0.5, n_customers),
    "Churn_Risk": (0.8 * (1 - rng.beta(2,5,n_customers)) + 0.2 * rng.normal(0,0.05,n_customers)).clip(0,1)
})

# tweak correlations to make the matrix interesting (introduce some dependencies)
data["Avg_Spend"] += 0.8 * (data["Visits"] - data["Visits"].mean()) * 1.5 / max(1, data["Visits"].std())
data["Session_Duration"] += 0.5 * (data["Pages_Per_Session"] - data["Pages_Per_Session"].mean())
data["Repeat_Rate"] = (0.4 * (data["Visits"] / (data["Visits"].max()+1)) + 0.6 * data["Repeat_Rate"]).clip(0,1)

# -------------------------
# 2. Compute correlation matrix
# -------------------------
corr = data.corr()

# -------------------------
# 3. Styling and plotting
# -------------------------
sns.set_style("white")
sns.set_context("talk", font_scale=0.9)

plt.figure(figsize=(8, 8))   # 8 in * 64 dpi => 512 pixels
cmap = sns.diverging_palette(220, 20, as_cmap=True)  # professional diverging palette

ax = sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    linewidths=0.8,
    linecolor="white",
    cmap=cmap,
    vmin=-1,
    vmax=1,
    square=True,
    cbar_kws={"shrink": 0.75, "label": "Pearson r"}
)

ax.set_title("Customer Engagement Correlation Matrix — Hickle Zieme & Powlowski", pad=16, fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

# -------------------------
# 4. Save PNG exactly 512x512
# -------------------------
plt.savefig("chart.png", dpi=64, bbox_inches="tight")  # 8*64 = 512 px
plt.close()
