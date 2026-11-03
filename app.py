
from pathlib import Path

import joblib
import pandas as pd

from flask import Flask, render_template, request


# -----------------------------------
# 1. Initialize Flask application
# -----------------------------------

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"


# -----------------------------------
# 2. Load trained ML models
# -----------------------------------

early_model = joblib.load(
    ARTIFACT_DIR / "early_warning_model.pkl"
)

classifier = joblib.load(
    ARTIFACT_DIR / "random_forest_classifier.pkl"
)

features = joblib.load(
    ARTIFACT_DIR / "early_warning_features.pkl"
)

default_profile = joblib.load(
    ARTIFACT_DIR / "default_student_profile.pkl"
)


# -----------------------------------
# 3. Create student input profile
# -----------------------------------

def prepare_student_data(form_data):

    student = default_profile.copy()

    for feature in features:

        if feature not in form_data:
            continue

        value = form_data.get(feature, "").strip()

        if value == "":
            continue

        original_value = default_profile[feature]

        try:
            if isinstance(original_value, (int, float)):
                student[feature] = float(value)
            else:
                student[feature] = value

        except ValueError:
            continue

    return pd.DataFrame(
        [student],
        columns=features
    )


# -----------------------------------
# 4. Identify possible weak areas
# -----------------------------------

def detect_weak_areas(student):

    weak_areas = []

    if student["studytime"] <= 2:
        weak_areas.append(
            "Study time is low. Try following a consistent study timetable."
        )

    if student["absences"] > 10:
        weak_areas.append(
            "Absences are relatively high. Improve class attendance."
        )

    if student["failures"] > 0:
        weak_areas.append(
            "Previous class failures may indicate a need for extra academic support."
        )

    if not weak_areas:
        weak_areas.append(
            "No major concerns detected by these basic indicators. "
            "Continue your study routine."
        )

    return weak_areas


# -----------------------------------
# 5. Home page
# -----------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# -----------------------------------
# 6. Prediction route
# -----------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        student_data = prepare_student_data(request.form)

        # Predict final grade out of 20
        predicted_grade = float(
            early_model.predict(student_data)[0]
        )

        predicted_grade = max(
            0,
            min(20, predicted_grade)
        )

        # Convert grade to percentage
        predicted_percentage = predicted_grade * 5

        # Predict PASS / FAIL
        predicted_result = int(
            classifier.predict(student_data)[0]
        )

        result = "PASS" if predicted_result == 1 else "FAIL"

        # Identify possible weak areas
        weak_areas = detect_weak_areas(
            student_data.iloc[0]
        )

        return render_template(
            "index.html",
            prediction_done=True,
            predicted_grade=round(predicted_grade, 2),
            predicted_percentage=round(predicted_percentage, 2),
            result=result,
            weak_areas=weak_areas
        )

    except Exception as error:

        return render_template(
            "index.html",
            error=str(error)
        )


# -----------------------------------
# 7. Run application locally
# -----------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )