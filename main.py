import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import tempfile
import csv
import os
from datetime import datetime

# ----------------------
# (Optional) Path to uploaded notebook (provided in session)
# ----------------------
NOTEBOOK_PATH = "/mnt/data/Heart Disease Predictions.ipynb"

# ----------------------
# Load trained model
# ----------------------
# Make sure heart_model.pkl is in the same folder as this script or provide the path
model = joblib.load("heart_model.pkl")

# ----------------------
# Helper: Log patient predictions for Admin Dashboard
# ----------------------
def log_patient_data(disease_prob, healthy_prob, user_data):
    file_path = "patient_logs.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Timestamp", "Age", "BP", "Cholesterol", "Sugar",
                "HeartRate", "HealthyProb", "DiseaseProb"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_data["Age"],
            user_data["Resting BP"],
            user_data["Cholesterol"],
            user_data["Fasting Sugar"],
            user_data["Max Heart Rate"],
            round(healthy_prob, 4),
            round(disease_prob, 4)
        ])


# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="❤️ Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Prediction App")
st.markdown("Fill in the details below to predict the **chance of heart disease** and get personalized suggestions.")

# Provide link to uploaded notebook (developer-specified local path)
st.sidebar.markdown("### Resources")
st.sidebar.markdown(f"- Notebook: `{NOTEBOOK_PATH}`")

# ----------------------
# Input fields
# ----------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=40)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.slider("Fasting Blood Sugar (mg/dl)", min_value=70, max_value=200, value=100)

with col2:
    restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
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
# Normal ranges for graph
# ----------------------
normal_ranges = {
    "Age": 40,
    "Resting BP": 120,
    "Cholesterol": 200,
    "Fasting Sugar": 100,
    "Max Heart Rate": 170
}

user_data = {
    "Age": age,
    "Resting BP": trestbps,
    "Cholesterol": chol,
    "Fasting Sugar": fbs,
    "Max Heart Rate": thalach,
}

# ----------------------
# Recommendations
# ----------------------
def get_recommendations(disease_prob, age, chol, trestbps, fbs, chest_pain, exang):
    recs = {"Medicine": [], "Exercise": [], "Lifestyle": []}

    if 30 <= disease_prob <= 50:
        recs["Medicine"].append("Consider preventive medicines (consult doctor).")
        if trestbps > 130:
            recs["Medicine"].append("BP control medication may be required.")
        if chol > 220:
            recs["Medicine"].append("Statins may be needed.")
        recs["Exercise"].append("Aerobic activities like walking or cycling.")
        recs["Lifestyle"].append("Reduce salt, sugar, and fried foods.")

    elif disease_prob > 50:
        recs["Medicine"].append("Prescription medicines likely required.")
        if trestbps > 140:
            recs["Medicine"].append("Strong BP medicines may be needed.")
        if chol > 240:
            recs["Medicine"].append("Cholesterol-lowering statins recommended.")
        if fbs > 120:
            recs["Medicine"].append("Diabetes medicines may be required.")
        recs["Exercise"].append("Light walking, yoga. Avoid heavy workouts.")
        recs["Lifestyle"].append("Strict diet, avoid smoking/alcohol.")

    return recs

# ----------------------
# HTML Report Generator
# ----------------------
def generate_html_report(disease_prob, healthy_prob, recs):
    html = f"""
    <h2>❤️ Heart Disease Prediction Report</h2>
    <h3>📊 Prediction Summary</h3>
    <p><b>Chance of Heart Disease:</b> {disease_prob:.2f}%</p>
    <p><b>Chance of Being Healthy:</b> {healthy_prob:.2f}%</p>
    <h3>🩺 Recommendations</h3>
    """
    for section, items in recs.items():
        html += f"<h4>{section}</h4><ul>"
        for i in items:
            html += f"<li>{i}</li>"
        html += "</ul>"
    return html

# ----------------------
# MAIN PREDICT BUTTON
# ----------------------
if st.button("🔍 Predict"):
    proba = model.predict_proba(features)[0]
    disease_prob = proba[1] * 100
    healthy_prob = proba[0] * 100

    st.subheader("📊 Prediction Result")
    st.write(f"**Chance of Heart Disease:** {disease_prob:.2f}%")
    st.write(f"**Chance of Being Healthy:** {healthy_prob:.2f}%")

    # Log this prediction for admin analytics
    try:
        log_patient_data(disease_prob, healthy_prob, user_data)
    except Exception as e:
        st.warning(f"Could not log patient data: {e}")

    # ----------------------
    # Probability Chart
    # ----------------------
    chart_data = pd.DataFrame({
        "Category": ["Healthy", "Disease"],
        "Probability": [healthy_prob, disease_prob]
    })

    fig = px.bar(
        chart_data,
        x="Category",
        y="Probability",
        color="Category",
        color_discrete_map={"Healthy": "green", "Disease": "red"},
        height=300
    )

    fig.update_layout(yaxis_title="Probability (%)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------
    # User vs Normal Comparison Chart
    # ----------------------
    st.subheader("📈 Your Health vs Normal Ranges")

    compare_df = pd.DataFrame({
        "Parameter": list(user_data.keys()),
        "User Value": list(user_data.values()),
        "Normal Value": list(normal_ranges.values())
    })

    fig2 = px.bar(
        compare_df,
        x="Parameter",
        y=["User Value", "Normal Value"],
        barmode="group",
        height=350
    )

    fig2.update_layout(yaxis_title="Value")
    st.plotly_chart(fig2, use_container_width=True)

    # ----------------------
    # Risk Messages & Recommendations
    # ----------------------
    recs = get_recommendations(disease_prob, age, chol, trestbps, fbs, chest_pain, exang)

    if disease_prob < 30:
        st.success("✅ Low risk! Keep up your healthy lifestyle.")
    elif 30 <= disease_prob <= 50:
        st.warning("⚠️ Moderate risk. Follow preventive measures.")
    else:
        st.error("🚨 High risk! Consult a cardiologist immediately.")

    st.subheader("🩺 Recommendations")
    for section, items in recs.items():
        if items:
            st.write(f"### {section}")
            for it in items:
                st.write(f"- {it}")

    # ----------------------
    # HTML Report Download
    # ----------------------
    report_html = generate_html_report(disease_prob, healthy_prob, recs)

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    with open(temp.name, "w", encoding="utf-8") as f:
        f.write(report_html)

    st.download_button(
        "📄 Download Health Report (HTML)",
        data=open(temp.name, "rb"),
        file_name="Heart_Report.html",
        mime="text/html"
    )

    # ---------------------------------------------
    # DOCTOR DASHBOARD (Cardiology Specialists)
    # ---------------------------------------------
    st.markdown("---")
    st.header("🩺 Meet Our Cardiologists")

    doctors = [
        {
            "name": "Dr. Amit Mittal",
            "degree": "MBBS, MD, DM (Cardiology)",
            "experience": "19+ Years Experience",
            "phone": "+91 8069305511",
            "email": "amit.mittal@cardiology.com",
            "hospital": "Apollo Hospital, Delhi",
            "image": "https://images.apollo247.in/doctors/5f02bdd3-b7ae-4a51-b1e0-ff2266b75f0f.jpg?tr=w-150,c-at_max,f-auto,q=80,dpr-2"
        },
        {
            "name": "Dr. Noopur Goyal",
            "degree": "M.D. Pediatrics, MBBS",
            "experience": "9+ Years Experience",
            "phone": "+91 8800447777",
            "email": "noopur.goyal@cardiologist.com",
            "hospital": "Yatharth Hospital, Greater Noida (Near Pari Chowk)",
            "image": "https://www.yatharthhospitals.com/uploads/doctor/dr-noopur-goyal45829537.jpg"
        },
        {
            "name": "Prof. Dr. Vivek Gupta",
            "degree": "MD, DM, FESC, FSCAI, FICC",
            "experience": "15+ Years Experience",
            "phone": "+91 8069305511",
            "email": "vivek.gupta@medilife.com",
            "hospital": "Apollo Hospital, Delhi",
            "image": "https://images.apollo247.in/doctors/f0c13537-efb6-452f-afef-32894abcb1cc-1738335734988.png?tr=w-150,c-at_max,f-auto,q=80,dpr-2"
        }
    ]

    cols = st.columns(3)
    for idx, doc in enumerate(doctors):
        with cols[idx]:
            st.image(doc["image"], width=220)
            st.subheader(doc["name"])
            st.write(f"**Degree:** {doc['degree']}")
            st.write(f"**Experience:** {doc['experience']}")
            st.write(f"**Hospital:** {doc['hospital']}")
            st.write(f"**Phone:** {doc['phone']}")
            st.write(f"**Email:** {doc['email']}")
            st.markdown("---")

# ----------------------
# ADMIN DASHBOARD (Sidebar toggle)
# ----------------------
st.sidebar.markdown("---")
admin_panel = st.sidebar.checkbox("🔐 Open Admin Dashboard")

if admin_panel:
    st.title("🏥 Admin Dashboard")

    st.markdown("### 📌 Overview of App Usage")

    if os.path.exists("patient_logs.csv"):
        df = pd.read_csv("patient_logs.csv")

        # Convert to numeric (in case)
        df["DiseaseProb"] = pd.to_numeric(df["DiseaseProb"], errors="coerce")
        df["HealthyProb"] = pd.to_numeric(df["HealthyProb"], errors="coerce")

        # --- Stats ---
        total_patients = len(df)
        high_risk_count = len(df[df["DiseaseProb"] > 50])
        moderate_risk = len(df[(df["DiseaseProb"] >= 30) & (df["DiseaseProb"] <= 50)])

        colA, colB, colC = st.columns(3)
        colA.metric("Total Patients Tested", total_patients)
        colB.metric("High-Risk Patients", high_risk_count)
        colC.metric("Moderate-Risk Patients", moderate_risk)

        # --- Daily usage chart ---
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Date"] = df["Timestamp"].dt.date
        usage = df.groupby("Date").size().reset_index(name="Count")

        st.markdown("### 📅 Daily Usage Statistics")
        fig_usage = px.line(usage, x="Date", y="Count", markers=True, title="Daily App Usage")
        st.plotly_chart(fig_usage, use_container_width=True)

        # --- Risk distribution ---
        st.markdown("### ⚠️ Patient Risk Distribution")
        fig_risk = px.histogram(df, x="DiseaseProb", nbins=20, title="Risk Score Distribution")
        st.plotly_chart(fig_risk, use_container_width=True)

        # --- Show raw logs ---
        st.markdown("### 📄 Patient Log Records")
        st.dataframe(df, use_container_width=True)

        # --- Download CSV ---
        st.download_button(
            label="⬇ Download Patient Logs (CSV)",
            data=df.to_csv(index=False),
            file_name="patient_logs.csv",
            mime="text/csv"
        )

    else:
        st.warning("No patient data found. Run at least 1 prediction.")
