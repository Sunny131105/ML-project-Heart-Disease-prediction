import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px

# ----------------------
# Load trained model
# ----------------------
model = joblib.load("heart_model.pkl")

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="❤️ Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Prediction App")
st.markdown("Fill in the details below to predict the **chance of heart disease** and get personalized suggestions.")

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
    thalach = st.number_input("Heart Beat Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# ----------------------
# Encoding categorical variables
# ----------------------
sex_map = {"Male": 1, "Female": 0}
cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
restecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
exang_map = {"Yes": 1, "No": 0}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

# ----------------------
# Feature vector (13 inputs now)
# ----------------------
features = np.array([[
    age,
    sex_map[sex],
    cp_map[chest_pain],
    trestbps,
    chol,
    1 if fbs > 120 else 0,
    restecg_map[restecg],
    thalach,
    exang_map[exang],
    oldpeak,
    slope_map[slope],
    ca,
    thal_map[thal]
]])


# ----------------------
# Function to generate recommendations
# ----------------------
def get_recommendations(age, chol, trestbps, fbs, chest_pain, exang):
    recommendations = {"Medicine": [], "Exercise": []}

    # Blood Pressure related
    if trestbps > 140:
        recommendations["Medicine"].append("Blood pressure control medicines (e.g., ACE inhibitors, Beta-blockers)")
        recommendations["Exercise"].append("Light yoga, daily walking (20–30 min), avoid heavy lifting")

    # Cholesterol related
    if chol > 240:
        recommendations["Medicine"].append("Statins to lower cholesterol")
        recommendations["Exercise"].append("Aerobic exercise like brisk walking, cycling, swimming")

    # Diabetes risk (high fasting blood sugar)
    if fbs > 120:
        recommendations["Medicine"].append("Sugar control medicines (e.g., Metformin)")
        recommendations["Exercise"].append("Daily 30 min brisk walk, strength training twice a week")

    # Chest pain risk
    if chest_pain in ["Typical Angina", "Atypical Angina"]:
        recommendations["Medicine"].append("Nitroglycerin or other angina-relief medicines")
        recommendations["Exercise"].append("Gentle exercises under medical supervision")

    # Exercise induced angina
    if exang == "Yes":
        recommendations["Exercise"].append("Avoid intense workouts, prefer light walking and breathing exercises")

    # Age-specific
    if age > 60:
        recommendations["Exercise"].append("Low impact exercises like tai chi, yoga, water aerobics")

    return recommendations


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

    # ✅ Plotly bar chart with custom colors
    chart_data = pd.DataFrame({
        "Category": ["Healthy", "Disease"],
        "Probability (%)": [healthy_prob, disease_prob],
        "Color": ["green", "red"]
    })

    fig = px.bar(
        chart_data,
        x="Category",
        y="Probability (%)",
        color="Color",
        color_discrete_map={"green": "green", "red": "red"},
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display messages and recommendations
    if disease_prob > 50:
        st.error("⚠️ High chance of heart disease. Please consult a doctor.")

        recs = get_recommendations(age, chol, trestbps, fbs, chest_pain, exang)

        st.subheader("💊 Suggested Medicines (General)")
        for med in recs["Medicine"]:
            st.write(f"- {med}")

        st.subheader("🏃 Suggested Exercises & Lifestyle")
        for ex in recs["Exercise"]:
            st.write(f"- {ex}")

    else:
        st.success("✅ Low chance of heart disease. Keep maintaining a healthy lifestyle!")

