"""
RegenPredict - app.py
MathXplore 2026 | Flask Prediction API
Run: python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, json, numpy as np

app = Flask(__name__)
CORS(app)

try:
    linear_model  = pickle.load(open("linear_model.pkl",  "rb"))
    poly_model    = pickle.load(open("poly_model.pkl",    "rb"))
    poly_features = pickle.load(open("poly_features.pkl", "rb"))
    scaler        = pickle.load(open("scaler.pkl",        "rb"))
    with open("coefficients.json") as f:
        coeff_data = json.load(f)
    print("✔ All models loaded.")
except FileNotFoundError as e:
    print(f"✘ {e} — Run model.py first!")
    exit(1)

# Features: regen_intensity, BRK, CH, SPD, CUR
FEATURES = ["regen_intensity", "BRK", "CH", "SPD", "CUR"]

def stress_label(s):
    if s < 10:  return {"label": "Low Stress",      "color": "#22c55e"}
    if s < 30:  return {"label": "Moderate Stress", "color": "#f59e0b"}
    if s < 60:  return {"label": "High Stress",     "color": "#ef4444"}
    return              {"label": "Critical Stress", "color": "#991b1b"}

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "project": "RegenPredict — MathXplore 2026", "endpoint": "/predict"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json()
        brk = float(d.get("BRK", 0))
        ch  = float(d.get("CH",  60))
        spd = float(d.get("SPD", 50))
        cur = float(d.get("CUR", 20))
        regen = abs(cur) if cur < 0 else 0.0

        X = np.array([[regen, brk, ch, spd, cur]])
        X_s = scaler.transform(X)

        pred_lin  = round(float(np.clip(linear_model.predict(X_s)[0], 0, 226)), 3)
        pred_poly = round(float(np.clip(poly_model.predict(poly_features.transform(X_s))[0], 0, 226)), 3)
        health    = stress_label(pred_lin)

        return jsonify({
            "battery_stress":      pred_lin,
            "battery_stress_poly": pred_poly,
            "unit":         "stress index",
            "health_status": health["label"],
            "color":         health["color"],
            "model_r2":      coeff_data["metrics"]["linear_r2"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/coefficients", methods=["GET"])
def coefficients():
    return jsonify(coeff_data)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  RegenPredict API — MathXplore 2026")
    print("  http://localhost:5000/predict")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
