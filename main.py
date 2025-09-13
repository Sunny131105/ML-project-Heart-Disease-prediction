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
# Feature vector
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
def get_recommendations(disease_prob, age, chol, trestbps, fbs, chest_pain, exang):
    recs = {"Medicine": [], "Exercise": [], "Lifestyle": []}

    # MODERATE RISK (30–50%)
    if 30 <= disease_prob <= 50:
        recs["Medicine"].append("Consider preventive medicines (consult doctor).")
        if trestbps > 130:
            recs["Medicine"].append("Mild BP control medication may be required.")
        if chol > 220:
            recs["Medicine"].append("Statins may be suggested for cholesterol control.")
        recs["Exercise"].append("Aerobic activities like cycling, brisk walking, swimming.")
        recs["Lifestyle"].append("Reduce salt, fried food, and sugary drinks.")

    # HIGH RISK (>50%)
    elif disease_prob > 50:
        recs["Medicine"].append("Prescription medicines likely required (consult cardiologist).")
        if trestbps > 140:
            recs["Medicine"].append("Strong BP medicines (ACE inhibitors, Beta-blockers).")
        if chol > 240:
            recs["Medicine"].append("Statins for cholesterol lowering.")
        if fbs > 120:
            recs["Medicine"].append("Diabetes control medicines (e.g., Metformin).")
        if chest_pain in ["Typical Angina", "Atypical Angina"]:
            recs["Medicine"].append("Angina medicines (Nitroglycerin).")

        recs["Exercise"].append("Gentle walking, yoga, tai chi (avoid heavy exercise).")
        if exang == "Yes":
            recs["Exercise"].append("Avoid intense workouts, prefer breathing exercises.")

        recs["Lifestyle"].append("Strict low-salt, low-oil diet with more fruits & vegetables.")
        recs["Lifestyle"].append("Avoid smoking & alcohol. Regular checkups every 3–6 months.")

    return recs


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

    # ✅ Plotly bar chart
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

    # ----------------------
    # Show risk messages
    # ----------------------
    if disease_prob < 30:
        st.success("✅ Low risk! Keep up your healthy lifestyle.")
    elif 30 <= disease_prob <= 50:
        st.warning("⚠️ Moderate risk. Follow preventive measures and monitor health.")
    else:
        st.error("🚨 High risk! Please consult a cardiologist immediately.")

    # ----------------------
    # Show recommendations ONLY if risk exists (>=30%)
    # ----------------------
    if disease_prob >= 30:
        recs = get_recommendations(disease_prob, age, chol, trestbps, fbs, chest_pain, exang)

        if recs["Medicine"]:
            st.subheader("💊 Medicine Suggestions (General)")
            for med in recs["Medicine"]:
                st.write(f"- {med}")

        if recs["Exercise"]:
            st.subheader("🏃 Exercise Suggestions")
            for ex in recs["Exercise"]:
                st.write(f"- {ex}")

        if recs["Lifestyle"]:
            st.subheader("🥗 Lifestyle & Diet Tips")
            for tip in recs["Lifestyle"]:
                st.write(f"- {tip}")


