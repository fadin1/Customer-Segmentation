import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Create sample customer data
data = {
    "CustomerID": list(range(1, 21)),
    "AnnualIncome": [
        15, 18, 20, 22, 25,
        35, 38, 40, 42, 45,
        55, 58, 60, 62, 65,
        75, 78, 80, 85, 90
    ],
    "SpendingScore": [
        80, 85, 78, 90, 82,
        40, 45, 35, 50, 42,
        60, 65, 55, 70, 62,
        20, 25, 15, 30, 22
    ]
}

df = pd.DataFrame(data)

# 2. Select features for clustering
X = df[["AnnualIncome", "SpendingScore"]]

# 3. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply K-Means clustering
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# 5. Display customer groups
print("\nCustomer Segmentation Results:")
print(df.to_string(index=False))

print("\nNumber of customers in each cluster:")
print(df["Cluster"].value_counts().sort_index())

# 6. Visualize the clusters
sns.scatterplot(
    data=df,
    x="AnnualIncome",
    y="SpendingScore",
    hue="Cluster",
    palette="Set1",
    s=120
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Annual Income (in thousands)")
plt.ylabel("Spending Score")
plt.legend(title="Customer Cluster")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()