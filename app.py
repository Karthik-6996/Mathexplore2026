"""
RegenPredict - app.py
MathXplore 2026 | Flask Prediction API
Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle, json, numpy as np
import os, requests as http_requests
from dotenv import load_dotenv
load_dotenv()

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

def local_project_answer(user_text):
    q = (user_text or "").strip().lower()
    if "r2" in q or "r²" in q or "accuracy" in q:
        return (
            "The linear model has R2 = 0.8093, which means it explains about 80.93% "
            "of battery stress variation in the dataset."
        )
    if "current" in q and ("importance" in q or "top" in q):
        return (
            "Current Draw is the dominant driver (57.0% importance), so changes in load/"
            "draw have the strongest impact on Battery Stress Index."
        )
    if "durbin" in q:
        return "Durbin-Watson is 2.006, which indicates no significant residual autocorrelation."
    if "stochastic" in q:
        return "The stochastic extension is BSI(t) = f(x_t) + e_t with e_t ~ N(0, sigma^2)."
    if "battery stress" in q or "bsi" in q:
        return (
            "Battery Stress Index (BSI) is modeled from regen intensity, braking force, "
            "SOC, speed, and current draw using multiple linear regression."
        )
    if "regenerative braking" in q:
        return (
            "Regenerative braking converts vehicle kinetic energy into electrical energy "
            "during deceleration, sending current back to the battery."
        )
    return (
        "I can still help with this project even though the external AI provider is unavailable. "
        "Ask about model metrics, feature importance, assumptions, or EV battery behavior."
    )

def fallback_chat_response(user_message, provider_error):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": local_project_answer(user_message),
                }
            }
        ],
        "fallback": True,
        "provider_error": provider_error,
    }

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

@app.route("/api/debug", methods=["GET"])
def debug_check():
    groq_key = os.getenv("GROQ_API_KEY")
    grok_key = os.getenv("GROK_API_KEY")
    key = groq_key or grok_key
    return jsonify({
        "GROQ_API_KEY_set": bool(groq_key),
        "GROK_API_KEY_set": bool(grok_key),
        "key_preview": (key[:8] + "..." + key[-4:]) if key and len(key) > 12 else ("too_short" if key else "MISSING"),
        "key_length": len(key) if key else 0,
    })

@app.route("/api/chat", methods=["POST"])
@app.route("/chat", methods=["POST"])
def chat():
    try:
        payload = request.get_json(silent=True) or {}
        messages = payload.get("messages", [])
        if not isinstance(messages, list) or len(messages) == 0:
            return jsonify({"error": "messages must be a non-empty array"}), 400

        user_message = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_message = str(msg.get("content", ""))
                break

        api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
        if not api_key:
            return jsonify(fallback_chat_response(user_message, "Missing GROQ_API_KEY (or GROK_API_KEY)")), 200

        resp = http_requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "RegenPredict/1.0",
            },
            json={
                "model": payload.get("model", "llama-3.3-70b-versatile"),
                "max_tokens": int(payload.get("max_tokens", 512)),
                "messages": messages,
            },
            timeout=25,
        )

        if resp.status_code == 200:
            return jsonify(resp.json())

        try:
            provider_error = resp.json().get("error", resp.text)
        except Exception:
            provider_error = f"Groq API HTTP {resp.status_code}"
        return jsonify(fallback_chat_response(user_message, str(provider_error))), 200
    except Exception as e:
        return jsonify(fallback_chat_response(user_message, str(e))), 200

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  RegenPredict — MathXplore 2026")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)