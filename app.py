
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model safely (Render-compatible path)
model_path = os.path.join(os.path.dirname(__file__), "final_diabetes_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Landing Page
@app.route("/")
def landing():
    return render_template("landing.html")

# Assessment Page
@app.route("/assessment")
def assessment():
    return render_template("index.html")

# Prediction Logic
@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = np.array([[
            float(request.form.get("Pregnancies")),
            float(request.form.get("Glucose")),
            float(request.form.get("BloodPressure")),
            float(request.form.get("SkinThickness")),
            float(request.form.get("Insulin")),
            float(request.form.get("BMI")),
            float(request.form.get("DiabetesPedigreeFunction")),
            float(request.form.get("Age"))
        ]])

        prediction = model.predict(input_data)[0]
        probability = float(model.predict_proba(input_data)[0][1]) * 100

        if prediction == 1:
            result = f"High Risk of Diabetes ({probability:.2f}%)"
            risk_class = "high"
            bar_color = "#dc2626"
        else:
            result = f"Low Risk of Diabetes ({probability:.2f}%)"
            risk_class = "low"
            bar_color = "#16a34a"

        return render_template(
            "index.html",
            prediction_text=result,
            probability=round(probability, 2),
            risk_class=risk_class,
            bar_color=bar_color
        )

    except Exception:
        return render_template(
            "index.html",
            prediction_text="Invalid input. Please check entered values."
        )

# Do NOT use debug mode in production
if __name__ == "__main__":
    app.run()
