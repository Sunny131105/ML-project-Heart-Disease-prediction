# 🫀 Heart Disease Prediction

🔗 **Live Demo:** https://ml-project-heart-disease-prediction-twhs2vajebmxt63axanerm.streamlit.app/

A **Machine Learning–powered web application** that predicts the **likelihood of heart disease** using clinical health parameters. Built with **Python**, **Scikit-Learn**, and **Streamlit** for an interactive user interface.

---

## 🧠 Project Overview  

Heart disease is one of the leading causes of death worldwide. This project aims to help users **quickly estimate their risk of heart disease** by entering basic medical and demographic information. The model uses trained machine learning algorithms on health data to provide a prediction based on user inputs.

---

## 📊 Features  

✔ Simple and intuitive **Streamlit web interface**  
✔ Supports input of key clinical and health attributes  
✔ Predicts heart disease likelihood in real time  
✔ Deployed online and accessible from any device  
✔ Useful for learning ML deployment and health analytics  

---

## 🛠️ Tech Stack  

- **Python** 🐍  
- **Scikit-Learn** (ML modeling)  
- **Pandas & NumPy** (data manipulation)  
- **Streamlit** (web app UI)  
- **Deployed on Streamlit Cloud**

---

## 📊 How It Works  

1. **Data Collection & Preprocessing**  
   - Health and clinical features are cleaned and prepped for model training.  
2. **Model Training**  
   - A supervised machine learning model is trained to classify high vs. low heart disease risk.  
3. **Web App Interface**  
   - Users fill in health metrics like age, blood pressure, cholesterol, etc.  
4. **Prediction**  
   - The app returns a prediction of whether the user is at risk of heart disease based on the model’s output.  
5. **Live Deployment**  
   - The app runs online so anyone can access it via the web link.

---



## 📥 Report Download & 📊 Graphical Analysis  

This application not only predicts the risk of heart disease but also provides **detailed analytical outputs** to help users better understand their health data.

### 📄 Downloadable Report  
- After submitting the input parameters, the app generates a **personalized health report**  
- The report can be **downloaded in CSV format**  
- Includes:
  - User input values
  - Prediction result (Heart Disease Risk / No Risk)
  - Comparison with standard normal health ranges  

This feature allows users to:
- Store their results for future reference  
- Share reports with healthcare professionals  
- Perform further offline analysis  

---

### 📊 Graphical View & Visualization  

The application provides **visual insights** to make the prediction more interpretable:

- 📈 **Comparison graphs** between:
  - User’s health metrics  
  - Normal/ideal health values

    
- 📈 Prediction Probability Visualization

This bar chart visually represents the probability distribution between:

🟢 Being Healthy

🔴 Having Heart Disease

![Prediction Result Graph](screenshots/prediction_result.png)


These graphs help users:
- Easily understand how their values differ from normal ranges  
- Gain better awareness of potential health risks  
- Make informed decisions based on visual data


- 📊 Health Parameters vs Normal Ranges

This comparative graph shows the user’s health values against normal medical ranges for key parameters such as:

• Age

• Resting Blood Pressure

• Cholesterol

• Fasting Blood Sugar

• Maximum Heart Rate

This visualization makes it easy to:

Identify deviations from normal health standards

Understand which parameters may contribute to higher risk

Gain better awareness through visual comparison

![Health vs Normal Ranges](screenshots/health_vs_normal.png)


---



## 🖥️ Usage  

To run this project locally:

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
