import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df_tt1 = pd.read_csv("term-test-1-result.csv")
df_tt2 = pd.read_csv("term-test-2-result.csv")
df_attendance_final = pd.read_csv("attendance-term-final.csv")

df_merged = pd.merge(df_tt1, df_tt2, on=["Registration Number", "Name"])
df_merged["Best Marks"] = df_merged[["TT-1 Marks", "TT-2 Marks"]].max(axis=1)
df_merged["Average Marks"] = df_merged[["TT-1 Marks", "TT-2 Marks"]].mean(axis=1)

df_merged.drop(columns=["TT-1 Marks", "TT-2 Marks"], inplace=True)
df_final = pd.merge(df_merged, df_attendance_final, on="Registration Number")

df_final["Final Marks"] = df_final["Term Final Marks"] * 0.7 + df_final["Average Marks"] + df_final["Attendance Marks"]
df_final.to_csv("final-result.csv", index=False)


X = df_final[["Final Marks"]]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_final["Cluster"] = kmeans.fit_predict(X)
plt.figure(figsize=(8, 5))
plt.scatter(df_final["Final Marks"], [0] * len(df_final), c=df_final["Cluster"], cmap="viridis", edgecolors="k")
plt.xlabel("Final Marks")
plt.title("K-Means Clustering of Final Marks")
plt.yticks([])  # Hide y-axis labels
plt.colorbar(label="Cluster")
plt.show()
