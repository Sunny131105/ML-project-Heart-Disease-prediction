import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ----------------------
# Load model (replace with your trained model .pkl if available)
# ----------------------
# Example: model = joblib.load("heart_model.pkl")
# For demo, we use random prediction
class DummyModel:
    def predict_proba(self, X):
        return np.array([[0.3, 0.7]])  # 70% chance of disease (demo)

model = DummyModel()

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="❤️ Health Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Prediction App")
st.markdown("Fill in the details below to predict the **chance of heart disease**.")

# ----------------------
# Input fields
# ----------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=40)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    )
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.slider("Fasting Blood Sugar (mg/dl)", min_value=70, max_value=200, value=100)

with col2:
    restecg = st.selectbox(
        "Resting ECG Results",
        ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"]
    )
    thalach = st.number_input(" Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("ST Segment", ["Upsloping", "Flat", "Downsloping"])
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# ----------------------
# Encoding categorical variables
# ----------------------
sex_map = {"Male": 1, "Female": 0}
cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}
restecg_map = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2
}
exang_map = {"Yes": 1, "No": 0}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

features = np.array([[
    age,
    sex_map[sex],
    cp_map[chest_pain],
    trestbps,
    chol,
    1 if fbs > 120 else 0,  # typical cutoff for fasting blood sugar
    restecg_map[restecg],
    thalach,
    exang_map[exang],
    oldpeak,
    slope_map[slope],
    thal_map[thal]
]])

# ----------------------
# Prediction
# ----------------------
if st.button("🔍 Predict"):
    proba = model.predict_proba(features)[0]
    disease_prob = proba[1] * 100
    healthy_prob = proba[0] * 100

    st.subheader("📊 Prediction Result")
    st.write(f"**Chance of Heart Disease:** {disease_prob:.2f}%")
    st.write(f"**Chance of Being Healthy:** {healthy_prob:.2f}%")

    # Matplotlib Bar Chart (3x3 inches)
    fig, ax = plt.subplots(figsize=(3, 3))
    categories = ["Healthy", "Disease"]
    values = [healthy_prob, disease_prob]
    ax.bar(categories, values, color=["green", "red"])
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)

    st.pyplot(fig)

    if disease_prob > 50:
        st.error("⚠️ High chance of heart disease. Please consult a doctor.")
    else:
        st.success("✅ Low chance of heart disease. Keep maintaining a healthy lifestyle!")
