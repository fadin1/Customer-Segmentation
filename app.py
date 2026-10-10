import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Customer Segmentation Dashboard")
st.write(
    "Analyze customer behavior using K-Means clustering "
    "and discover useful customer groups."
)

# -----------------------------
# Generate sample dataset
# -----------------------------
@st.cache_data
def generate_data(n=500):
    rng = np.random.default_rng(42)

    # Create four illustrative customer profiles
    profiles = [
        (125, 25, 24, 22, 3),   # Budget shoppers
        (125, 75, 42, 78, 12),  # Active spenders
        (125, 30, 38, 35, 5),   # Occasional shoppers
        (125, 80, 30, 88, 16)   # High spenders
    ]

    records = []

    for count, income_mean, age_mean, score_mean, freq_mean in profiles:
        for _ in range(count):
            records.append({
                "Age": np.clip(rng.normal(age_mean, 6), 18, 75),
                "Annual Income (k$)": np.clip(
                    rng.normal(income_mean, 10), 15, 150
                ),
                "Spending Score": np.clip(
                    rng.normal(score_mean, 10), 1, 100
                ),
                "Purchase Frequency": np.clip(
                    rng.normal(freq_mean, 2), 1, 30
                ),
                "Gender": rng.choice(["Female", "Male"])
            })

    df = pd.DataFrame(records)
    df["Age"] = df["Age"].round().astype(int)
    df["Annual Income (k$)"] = df["Annual Income (k$)"].round(1)
    df["Spending Score"] = df["Spending Score"].round(1)
    df["Purchase Frequency"] = df["Purchase Frequency"].round().astype(int)

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


df = generate_data()

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ Dashboard Settings")

n_clusters = st.sidebar.slider(
    "Number of clusters (K)",
    min_value=2,
    max_value=8,
    value=4
)

features = st.sidebar.multiselect(
    "Features for clustering",
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score",
        "Purchase Frequency"
    ],
    default=["Annual Income (k$)", "Spending Score"]
)

if len(features) < 2:
    st.warning("Please select at least two features.")
    st.stop()

# -----------------------------
# K-Means clustering
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

model = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

df["Cluster"] = model.fit_predict(X_scaled)

score = silhouette_score(
    X_scaled,
    df["Cluster"],
    sample_size=min(1000, len(df)),
    random_state=42
)

# -----------------------------
# KPI cards
# -----------------------------
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Customers", f"{len(df):,}")
c2.metric("Customer Segments", n_clusters)
c3.metric(
    "Average Income",
    f"${df['Annual Income (k$)'].mean():.1f}k"
)
c4.metric("Silhouette Score", f"{score:.3f}")

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔎 Cluster Analysis",
    "📋 Customer Data",
    "💡 Business Insights"
])

# -----------------------------
# Overview
# -----------------------------
with tab1:
    st.subheader("Customer Segmentation")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            df,
            x="Annual Income (k$)",
            y="Spending Score",
            color=df["Cluster"].astype(str),
            hover_data=["Age", "Gender", "Purchase Frequency"],
            title="Income vs Spending Score",
            labels={"color": "Cluster"},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df,
            x="Age",
            color=df["Cluster"].astype(str),
            barmode="overlay",
            opacity=0.7,
            title="Customer Age Distribution",
            labels={"color": "Cluster"}
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        df,
        x=df["Cluster"].astype(str),
        y="Purchase Frequency",
        color=df["Cluster"].astype(str),
        title="Purchase Frequency by Cluster",
        labels={"x": "Cluster"}
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Cluster analysis
# -----------------------------
with tab2:
    st.subheader("Compare Customer Segments")

    summary = df.groupby("Cluster").agg(
        Customers=("Age", "size"),
        Average_Age=("Age", "mean"),
        Average_Income=("Annual Income (k$)", "mean"),
        Average_Spending=("Spending Score", "mean"),
        Average_Purchases=("Purchase Frequency", "mean")
    ).round(2)

    st.dataframe(summary, use_container_width=True)

    fig = px.bar(
        summary.reset_index(),
        x="Cluster",
        y="Customers",
        color="Cluster",
        title="Number of Customers in Each Cluster",
        text="Customers"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Cluster IDs are arbitrary labels, not rankings. "
        "Interpret groups using their average characteristics."
    )

# -----------------------------
# Customer data
# -----------------------------
with tab3:
    st.subheader("Explore Customer Records")

    selected_cluster = st.selectbox(
        "Filter by cluster",
        ["All"] + list(range(n_clusters))
    )

    filtered = df.copy()

    if selected_cluster != "All":
        filtered = filtered[
            filtered["Cluster"] == selected_cluster
        ]

    st.write(f"Showing {len(filtered)} customers")
    st.dataframe(filtered, use_container_width=True)

    csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download customer data (CSV)",
        data=csv,
        file_name="customer_segments.csv",
        mime="text/csv"
    )

# -----------------------------
# Business insights
# -----------------------------
with tab4:
    st.subheader("Customer Segment Insights")

    insights = df.groupby("Cluster").agg(
        Customers=("Age", "size"),
        Income=("Annual Income (k$)", "mean"),
        Spending=("Spending Score", "mean"),
        Purchases=("Purchase Frequency", "mean")
    ).round(1)

    for cluster, row in insights.iterrows():
        st.markdown(f"### Cluster {cluster}")

        st.write(
            f"- Customers: {int(row['Customers'])} "
            f"({row['Customers'] / len(df) * 100:.1f}% of the dataset)"
        )
        st.write(f"- Average annual income: ${row['Income']:.1f}k")
        st.write(f"- Average spending score: {row['Spending']:.1f}/100")
        st.write(f"- Average purchase frequency: {row['Purchases']:.1f}")

        if row["Income"] >= 60 and row["Spending"] >= 60:
            st.info(
                "Possible high-value segment: consider loyalty rewards "
                "and personalized offers."
            )
        elif row["Income"] >= 60 and row["Spending"] < 40:
            st.info(
                "Possible under-engaged segment: test targeted offers "
                "to encourage purchases."
            )
        elif row["Spending"] >= 65:
            st.info(
                "High-spending segment: consider retention campaigns "
                "and relevant product recommendations."
            )
        else:
            st.info(
                "Explore product preferences and test promotions "
                "appropriate to this segment."
            )

st.divider()
st.caption(
    "Educational demo using synthetic data. "
    "Cluster interpretations are suggestions, not proven customer behavior."
)