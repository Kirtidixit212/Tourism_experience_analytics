import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load Data and Models
# -----------------------------

df = pd.read_csv("merged_tourism_dataset (1).csv")

reg_model = pickle.load(open("regression_model (1).pkl", "rb"))
clf_model = pickle.load(open("classification_model (1).pkl", "rb"))

# -----------------------------
# Build Recommendation System (Dynamic)
# -----------------------------

user_item = df.pivot_table(
    index="UserId",
    columns="Attraction",
    values="Rating",
    fill_value=0
)

similarity = cosine_similarity(user_item)

# -----------------------------
# Streamlit Page Config
# -----------------------------

st.set_page_config(page_title="Tourism Analytics App", layout="wide")

st.title("🌍 Tourism Experience Analytics System")
st.write("Predict Attraction Ratings, Visit Mode & Get Recommendations")

st.markdown("---")

# -----------------------------
# Sidebar Inputs
# -----------------------------

st.sidebar.header("User Input")

selected_continent = st.sidebar.selectbox(
    "Select Continent",
    sorted(df["Continent"].dropna().unique())
)

selected_month = st.sidebar.selectbox(
    "Select Visit Month",
    sorted(df["VisitMonth"].dropna().unique())
)

selected_attraction = st.sidebar.selectbox(
    "Select Attraction",
    sorted(df["Attraction"].dropna().unique())
)

# Add user selection for recommendation (IMPORTANT)
selected_user = st.sidebar.selectbox(
    "Select User for Recommendation",
    user_item.index
)

# -----------------------------
# Recommendation Function
# -----------------------------

def recommend_attractions(user_id, top_n=5):

    if user_id not in user_item.index:
        return []

    user_index = user_item.index.get_loc(user_id)
    similarity_scores = similarity[user_index]

    similar_users_idx = np.argsort(similarity_scores)[::-1][1:top_n+1]

    recommended = []

    for idx in similar_users_idx:
        liked_items = user_item.iloc[idx]
        high_rated = liked_items[liked_items > 4].index.tolist()
        recommended.extend(high_rated)

    return list(set(recommended))[:top_n]

# -----------------------------
# Prediction Button
# -----------------------------

if st.sidebar.button("Predict"):

    st.subheader("📊 Prediction Results")

    sample_row = df[df["Attraction"] == selected_attraction].iloc[0:1].copy()

    sample_row["VisitMonth"] = selected_month
    sample_row["Continent"] = selected_continent

    X_input = sample_row.drop(["Rating"], axis=1, errors="ignore")
    X_input = pd.get_dummies(X_input)

    X_input_reg = X_input.reindex(columns=reg_model.feature_names_in_, fill_value=0)
    X_input_clf = X_input.reindex(columns=clf_model.feature_names_in_, fill_value=0)

    rating_pred = reg_model.predict(X_input_reg)
    st.success(f"⭐ Predicted Rating: {round(rating_pred[0], 2)}")

    visit_mode_pred = clf_model.predict(X_input_clf)
    st.success(f"🧳 Predicted Visit Mode: {visit_mode_pred[0]}")

    # -----------------------------
    # Recommendation
    # -----------------------------

    st.subheader("🎯 Recommended Attractions")

    recommendations = recommend_attractions(selected_user)

    if recommendations:
        for rec in recommendations:
            st.write(f"• {rec}")
    else:
        st.write("No recommendations available.")

st.markdown("---")
st.write("Built with ❤️ using Streamlit & Machine Learning")