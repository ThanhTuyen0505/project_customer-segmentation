import streamlit as st
import pandas as pd
import joblib

# BASE_DIR = thư mục hiện tại của app
BASE_DIR = Path.cwd() / "demo"
# --- Load model, scaler, cluster profile ---
kmeans = joblib.load("BASE_DIR / kmeans_model.pkl")
scaler = joblib.load("BASE_DIR / scaler.pkl")
cluster_profile = joblib.load("BASE_DIR / cluster_profile.pkl")  # bảng trung bình các cluster

st.title("Customer Segmentation App")
st.write("Enter full customer details to predict the segment.")

# --- Demographic ---
age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
total_children = st.number_input("Total Children", min_value=0, max_value=10, value=0)

# --- Purchase Behavior ---
mnt_wines = st.number_input("Amount spent on Wines", min_value=0, max_value=5000, value=0)
mnt_fruits = st.number_input("Amount spent on Fruits", min_value=0, max_value=5000, value=0)
mnt_meat = st.number_input("Amount spent on Meat Products", min_value=0, max_value=5000, value=0)
mnt_fish = st.number_input("Amount spent on Fish Products", min_value=0, max_value=5000, value=0)
mnt_sweet = st.number_input("Amount spent on Sweet Products", min_value=0, max_value=5000, value=0)
mnt_gold = st.number_input("Amount spent on Gold Products", min_value=0, max_value=5000, value=0)

num_deals_purchases = st.number_input("Number of Deal Purchases", min_value=0, max_value=100, value=0)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=0)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=0)

recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)
marital_status = st.selectbox("Marital Status", options=["Single", "Married"])
education = st.selectbox("Education", options=["Undergraduate", "Postgraduate", "Graduate"])


# --- Prepare input dataframe ---
feature_cols = [
    "Age","Income","TotalChildren",
    "MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds",
    "NumDealsPurchases","NumWebPurchases","NumStorePurchases",
    "Recency","Marital_Status_Single","Education_Postgraduate","Education_Undergraduate"    
]

input_data = pd.DataFrame(columns=feature_cols)
input_data.loc[0] = [
    age,
    income,
    total_children,
    mnt_wines,
    mnt_fruits,
    mnt_meat,
    mnt_fish,
    mnt_sweet,
    mnt_gold,
    num_deals_purchases,
    num_web_purchases,
    num_store_purchases,
    recency,
    1 if marital_status == "Single" else 0,   # Marital_Status_Single
    1 if education == "Postgraduate" else 0,  # Education_Postgraduate
    1 if education == "Undergraduate" else 0, # Education_Undergraduate

]


# --- Scale input ---
input_scaled = scaler.transform(input_data)

# --- Predict ---
input_scaled=scaler.transform(input_data)
if st.button("Predict Segment"):
    cluster=kmeans.predict(input_scaled)[0]
    st.success(f"Predict Segment: Cluster {cluster}")
