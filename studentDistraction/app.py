from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("machine.pkl")

CSV_FILE = "students.csv"

print("Running from:", os.getcwd())
print("Model path:", os.path.abspath("machine.pkl"))
print("Model expects:", model.n_features_in_)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Incoming Data:", data)

        # =========================
        # SAFE FLOAT FUNCTION
        # =========================
        def safe_float(value):
            if value == "" or value is None:
                return 0.0
            return float(value)

        # =========================
        # INPUTS
        # =========================
        student_id = data.get("student_id", "unknown")

        study = safe_float(data.get("study_hours"))
        social = safe_float(data.get("social_media"))
        youtube = safe_float(data.get("youtube"))
        gaming = safe_float(data.get("gaming"))
        sleep = safe_float(data.get("sleep"))
        attendance = safe_float(data.get("attendance"))
        grade = safe_float(data.get("grade"))
        breaks = safe_float(data.get("breaks"))
        stress = safe_float(data.get("stress"))

        # =========================
        # MODEL INPUT (IMPORTANT)
        # =========================
        features = pd.DataFrame([{
            "study_hours_per_day": study,
            "social_media_hours": social,
            "youtube_hours": youtube,
            "gaming_hours": gaming,
            "sleep_hours": sleep,
            "attendance_percentage": attendance,
            "final_grade": grade,
            "breaks_per_day": breaks,
            "stress_level": stress
        }])

        print("Features Sent to Model:\n", features)

        # =========================
        # PREDICTION
        # =========================
        prediction = model.predict(features)[0]

        # =========================
        # RISK (PROBABILITY)
        # =========================
        risk = model.predict_proba(features)[0][1]  # probability of distracted
        risk_percent = round(risk * 100, 2)

        # Risk Label
        if risk_percent < 30:
            risk_label = "🟢 Low Risk"
        elif risk_percent < 70:
            risk_label = "🟡 Medium Risk"
        else:
            risk_label = "🔴 High Risk"

        # =========================
        # EXTRA LOGIC (SCORE)
        # =========================
        total_screen_time = social + youtube + gaming

        distraction_score = (
            (study < 4) +
            (total_screen_time > 6) +
            (sleep < 6) +
            (attendance < 75) +
            (grade < 50) +
            (breaks > 8) +
            (stress > 7)
        )

        # =========================
        # RESULT TEXT
        # =========================
        result = "🚨 Student is Highly Distracted" if prediction == 1 else "✅ Student is Focused"

        # =========================
        # SAVE DATA
        # =========================
        new_row = {
            "student_id": student_id,
            "study_hours_per_day": study,
            "social_media_hours": social,
            "youtube_hours": youtube,
            "gaming_hours": gaming,
            "sleep_hours": sleep,
            "attendance_percentage": attendance,
            "final_grade": grade,
            "breaks_per_day": breaks,
            "stress_level": stress,
            "prediction": int(prediction),
            "risk_percent": risk_percent,
            "distraction_score": int(distraction_score)
        }

        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

        df.to_csv(CSV_FILE, index=False)

        # =========================
        # RESPONSE
        # =========================
        return jsonify({
            "result": result,
            "score": int(distraction_score),
            "risk": risk_percent,
            "risk_label": risk_label
        })

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({
            "result": "⚠️ Backend Error: Check terminal"
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)