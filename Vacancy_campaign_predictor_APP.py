import streamlit as st
import joblib
import pandas as pd

# --- Load trained components from notebook ---
model = joblib.load("campaign_model.pkl")          # Trained logistic regression model
scaler = joblib.load("campaign_scaler.pkl")        # StandardScaler used in notebook
pca = joblib.load("campaign_pca.pkl")              # PCA used in notebook

# --- Streamlit App Title ---
st.title("Recruitment Campaign Success Estimator")

# --- Sliders for input ---
budget = st.slider("Total Budget (€)", 0, 2500, 1000, step=100)
weeks = st.slider("Campaign Duration (weeks)", 1, 12, 4)
meta_pct = st.slider("Meta %", 0, 100, 20, step=5)
indeed_pct = st.slider("Indeed %", 0, 100, 20, step=5)
google_pct = st.slider("Google Ads %", 0, 100, 20, step=5)
linkedin_pct = st.slider("LinkedIn %", 0, 100, 20, step=5)
youtube_pct = st.slider("YouTube %", 0, 100, 20, step=5)

# --- Check total allocation ---
sum_pct = meta_pct + indeed_pct + google_pct + linkedin_pct + youtube_pct
st.markdown(f"**Total allocation:** {sum_pct}%")
if sum_pct != 100:
    st.error("❌ Budget distribution must equal 100%")
    st.stop()

# --- Basic input range check (just like in notebook) ---
if budget < 750 or budget > 3000 or weeks < 2 or weeks > 8:
    st.warning("⚠️ Not enough data to make a reliable prediction. Try adjusting budget or duration.")
    st.stop()

# --- Prepare input features ---
input_data = pd.DataFrame([{
    'campaign_weeks': weeks,
    'total_add_budget': budget,
    'difficulty_level_num': 1,
    'meta_pct': meta_pct,
    'indeed_pct': indeed_pct,
    'linkedin_pct': linkedin_pct,
    'google_ads_pct': google_pct,
    'youtube_pct': youtube_pct,
    'sum_budget_pct': sum_pct
}])

# --- Ensure same feature order as during training ---
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

# --- Apply same preprocessing steps as in notebook ---
X_scaled = scaler.transform(input_data)
X_pca = pca.transform(X_scaled)

# --- Make prediction ---
prob = model.predict_proba(X_pca)[0][1]
label = "✅ Likely to get a qualified candidate" if prob >= 0.5 else "❌ Unlikely to succeed"

# --- Show result ---
st.subheader("Prediction")
st.markdown(label)
st.markdown(f"**Predicted success probability:** {round(prob * 100, 1)}%")
