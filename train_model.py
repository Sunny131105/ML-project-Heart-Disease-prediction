import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset (UCI Heart Disease dataset)
url = "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"
df = pd.read_csv("C:\\Users\\SUNNY SANGWAN\\Downloads\\heart (1).csv")

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "heart_model.pkl")

print("✅ Model trained and saved as heart_model.pkl")
