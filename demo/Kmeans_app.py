import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# ------------------------------
# Thư mục demo
# ------------------------------
BASE_DIR = Path.cwd() / "demo"

# ------------------------------
# Load model, scaler, cluster profile
# ------------------------------
kmeans = joblib.load(BASE_DIR / "kmeans_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
cluster_profile = joblib.load(BASE_DIR / "cluster_profile.pkl")

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Customer Segmentation App")
st.write("Enter full customer details to predict the segment.")

# Input fields
income = st.number_input("Income", min_value=0, max_value=120000, value=52000, step=1000)
num_deals_purchases = st.number_input("Number of Deal Purchases", min_value=0, max_value=6, value=2)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=12, value=4)
num_catalog_purchases = st.number_input("Number of Catalog Purchases", min_value=0, max_value=10, value=2)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=13, value=5)
num_web_visits_month = st.number_input("Number of Web Visits per Month", min_value=0, max_value=13, value=6)
total_children = st.number_input("Total Children", min_value=0, max_value=3, value=1)
total_spending = st.number_input("Total Spending", min_value=0, max_value=2200, value=400, step=50)
education = st.selectbox("Education", ["Graduate", "Undergraduate", "Postgraduate"])

# ------------------------------
# Prepare input as numpy array
# ------------------------------
input_array = np.array([[
    income,
    num_deals_purchases,
    num_web_purchases,
    num_catalog_purchases,
    num_store_purchases,
    num_web_visits_month,
    total_children,
    total_spending,
    1 if education == "Postgraduate" else 0,
    1 if education == "Undergraduate" else 0
]], dtype=float)

# ------------------------------
# Scale input
# ------------------------------
input_scaled = scaler.transform(input_array)

# ------------------------------
# Predict and show results
# ------------------------------
if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    st.success(f"Predicted Segment: Cluster {cluster}")
    st.subheader("Cluster Profile (Average Values)")
    st.dataframe(cluster_profile.loc[[cluster]])
