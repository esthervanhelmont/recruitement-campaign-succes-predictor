import streamlit as st
import pandas as pd
import pickle

with open("campaign_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("campaign_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("campaign_pca.pkl", "rb") as f:
    pca = pickle.load(f)

# --- App title ---
st.title("Recruitment Campaign Success Estimator")

# --- Budget and duration sliders ---
budget = st.slider("Total Budget (€)", 0, 2500, 1000, step=100)
weeks = st.slider("Campaign Duration (weeks)", 1, 12, 4)

# --- Difficulty level immediately after duration ---
difficulty_label = st.selectbox("Campaign Difficulty Level", ["Easy", "Medium", "Hard"], index=1)
difficulty_map = {"Easy": 0, "Medium": 1, "Hard": 2}
difficulty_level = difficulty_map[difficulty_label]

# --- Channel allocation sliders ---
meta_pct = st.slider("Meta %", 0, 100, 20, step=5)
indeed_pct = st.slider("Indeed %", 0, 100, 20, step=5)
google_pct = st.slider("Google Ads %", 0, 100, 20, step=5)
linkedin_pct = st.slider("LinkedIn %", 0, 100, 20, step=5)
youtube_pct = st.slider("YouTube %", 0, 100, 20, step=5)

# --- Allocation check ---
sum_pct = meta_pct + indeed_pct + google_pct + linkedin_pct + youtube_pct
st.markdown(f"**Total allocation:** {sum_pct}%")
if sum_pct != 100:
    st.error("❌ Budget distribution must equal 100%")
    st.stop()

# --- Basic validity checks ---
if budget < 750 or budget > 3000 or weeks < 2 or weeks > 8:
    st.warning("⚠️ Not enough data to make a reliable prediction. Try adjusting budget or duration.")
    st.stop()

# --- Input DataFrame in correct order ---
input_data = pd.DataFrame([{
    'campaign_weeks': weeks,
    'total_add_budget': budget,
    'difficulty_level_num': difficulty_level,
    'meta_pct': meta_pct,
    'indeed_pct': indeed_pct,
    'linkedin_pct': linkedin_pct,
    'google_ads_pct': google_pct,
    'youtube_pct': youtube_pct,
    'sum_budget_pct': sum_pct
}])

feature_order = [
    'campaign_weeks',
    'total_add_budget',
    'difficulty_level_num',
    'meta_pct',
    'indeed_pct',
    'linkedin_pct',
    'google_ads_pct',
    'youtube_pct',
    'sum_budget_pct'
]
input_data = input_data[feature_order]

# --- Preprocessing and prediction ---
X_scaled = scaler.transform(input_data)
X_pca = pca.transform(X_scaled)
prob = model.predict_proba(X_pca)[0][1]
label = "✅ Likely to get a qualified candidate" if prob >= 0.5 else "❌ Unlikely to succeed"

# --- Display result ---
st.subheader("Prediction")
st.markdown(label)
st.markdown(f"**Predicted success probability:** {round(prob * 100, 1)}%")
