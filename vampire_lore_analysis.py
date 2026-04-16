import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("vampire_lore_dataset.csv")

print("First 5 rows:")
print(df.head(), "\n")

print("Dataset shape:", df.shape, "\n")

print("Average victims per month:", round(df["victims_per_month"].mean(), 2))
print("Survival rate:", round(df["survived_story"].mean() * 100, 2), "%\n")

victims_by_region = (
    df.groupby("origin_region", as_index=False)["victims_per_month"]
    .mean()
    .sort_values("victims_per_month", ascending=False)
)
print("Average victims per month by region:")
print(victims_by_region, "\n")

survival_by_weakness = (
    df.groupby("weakness", as_index=False)["survived_story"]
    .mean()
    .sort_values("survived_story", ascending=False)
)
print("Survival rate by weakness:")
print(survival_by_weakness, "\n")

victims_by_century = (
    df.groupby("century", as_index=False)["victims_per_month"]
    .mean()
    .sort_values("century")
)
print("Average victims per month by century:")
print(victims_by_century, "\n")

transform_summary = (
    df.groupby("can_transform", as_index=False)["victims_per_month"]
    .mean()
)
transform_summary["can_transform"] = transform_summary["can_transform"].map({0:"No", 1:"Yes"})
print("Average victims per month by transformation ability:")
print(transform_summary, "\n")

plt.figure(figsize=(8, 5))
plt.bar(victims_by_region["origin_region"], victims_by_region["victims_per_month"])
plt.title("Average Victims per Month by Region")
plt.xlabel("Region")
plt.ylabel("Average Victims per Month")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("victims_by_region.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(survival_by_weakness["weakness"], survival_by_weakness["survived_story"])
plt.title("Survival Rate by Weakness")
plt.xlabel("Weakness")
plt.ylabel("Average Survival Rate")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("survival_by_weakness.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(df["victims_per_month"], df["survived_story"])
for _, row in df.iterrows():
    plt.annotate(row["vampire"], (row["victims_per_month"], row["survived_story"]), fontsize=7)
plt.title("Victims per Month vs Story Survival")
plt.xlabel("Victims per Month")
plt.ylabel("Survived Story")
plt.tight_layout()
plt.savefig("victims_vs_survival.png")
plt.close()

print("Analysis complete. Charts saved as PNG files.")