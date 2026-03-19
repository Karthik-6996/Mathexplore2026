"""
RegenPredict - app.py
MathXplore 2026 | Flask Prediction API
Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle, json, numpy as np
import os
from urllib import request as urlrequest, error as urlerror

app = Flask(__name__)
CORS(app)

# ── Load models ───────────────────────────────────────────────────────────────
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

FEATURES = ["regen_intensity", "BRK", "CH", "SPD", "CUR"]

def stress_label(s):
    if s < 10:  return {"label": "Low Stress",      "color": "#22c55e"}
    if s < 30:  return {"label": "Moderate Stress", "color": "#f59e0b"}
    if s < 60:  return {"label": "High Stress",     "color": "#ef4444"}
    return              {"label": "Critical Stress", "color": "#991b1b"}

# ── Serve dashboard ───────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    base = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base, "index.html")

@app.route("/<path:filename>", methods=["GET"])
def static_files(filename):
    base = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base, filename)

# ── API routes ────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        d     = request.get_json()
        brk   = float(d.get("BRK", 0))
        ch    = float(d.get("CH",  60))
        spd   = float(d.get("SPD", 50))
        cur   = float(d.get("CUR", 20))
        regen = abs(cur) if cur < 0 else 0.0

        X   = np.array([[regen, brk, ch, spd, cur]])
        X_s = scaler.transform(X)

        pred_lin  = round(float(np.clip(linear_model.predict(X_s)[0], 0, 226)), 3)
        pred_poly = round(float(np.clip(poly_model.predict(poly_features.transform(X_s))[0], 0, 226)), 3)
        health    = stress_label(pred_lin)

        return jsonify({
            "battery_stress":      pred_lin,
            "battery_stress_poly": pred_poly,
            "unit":                "stress index",
            "health_status":       health["label"],
            "color":               health["color"],
            "model_r2":            coeff_data["metrics"]["linear_r2"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/coefficients", methods=["GET"])
def coefficients():
    return jsonify(coeff_data)

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

@app.route("/api/chat", methods=["POST"])
@app.route("/chat", methods=["POST"])
def chat():
    try:
        payload = request.get_json(silent=True) or {}
        messages = payload.get("messages", [])
        if not isinstance(messages, list) or len(messages) == 0:
            return jsonify({"error": "messages must be a non-empty array"}), 400

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return jsonify({"error": "Server is missing GROQ_API_KEY"}), 500

        body = json.dumps({
            "model": payload.get("model", "llama-3.3-70b-versatile"),
            "max_tokens": int(payload.get("max_tokens", 512)),
            "messages": messages,
        }).encode("utf-8")

        req = urlrequest.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        with urlrequest.urlopen(req, timeout=25) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        return jsonify(result)
    except urlerror.HTTPError as e:
        raw_body = ""
        try:
            raw_body = e.read().decode("utf-8")
            err_payload = json.loads(raw_body)
        except Exception:
            msg = raw_body.strip() if raw_body else f"Groq API HTTP {e.code}"
            err_payload = {"error": msg}
        return jsonify(err_payload), e.code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  RegenPredict — MathXplore 2026")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)