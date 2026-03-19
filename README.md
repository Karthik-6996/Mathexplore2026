# RegenPredict — MathXplore 2026

> A mathematically validated regression model that quantifies how regenerative braking patterns drive battery stress in electric vehicles.

**Team:** Karthik G (NB25ISE104) · Disha (NB25CSE109) · Guru Raghav (NB25ISE078)  
**Institution:** Nitte Meenakshi Institute of Technology, Bengaluru  
**Competition:** MathXplore 2026 — Mathematical Modelling & Innovation  

---

## What This Project Does

Electric vehicles use regenerative braking to recover kinetic energy during deceleration. This project mathematically models how regenerative braking behaviour — intensity, frequency, speed, current draw, and state of charge — affects battery stress using **Multiple Linear Regression** trained on **24,277 real EV driving records**.

**Key Results:**
- Linear R² = **0.8093** (80.9% of battery stress variance explained)
- Polynomial R² = **1.0000**
- Dataset: 24,277 real EV telemetry samples with 6,624 logged regen events
- All 6 regression assumptions formally validated

---

## Project Structure

```
regen-braking-project/
├── model.py              # Data processing, model training, exports
├── app.py                # Flask REST API (localhost:5000)
├── index.html            # Interactive prediction dashboard
├── dataset.csv           # Real EV driving telemetry (24,277 samples)
├── coefficients.json     # Trained model coefficients + metrics (generated)
├── linear_model.pkl      # Saved linear regression model (generated)
├── poly_model.pkl        # Saved polynomial regression model (generated)
├── poly_features.pkl     # Polynomial feature transformer (generated)
├── scaler.pkl            # StandardScaler for input normalization (generated)
└── README.md             # This file
```

---

## Mathematical Model

**Deterministic Model — Multiple Linear Regression:**

```
BSI = β₀ + β₁·RegenIntensity + β₂·BrakingForce + β₃·SOC + β₄·Speed + β₅·CurrentDraw + ε
```

**Stochastic Extension:**

```
BSI(t) = f(xₜ) + εₜ,   εₜ ~ N(0, σ²)
```

**Variables:**

| Symbol | Variable | Description |
|--------|----------|-------------|
| BSI | Battery Stress Index | Target — electrochemical stress |
| β₁ · RegenI | Regen Intensity | Discharge current during braking (A) |
| β₂ · BRK | Braking Force | Pedal braking intensity |
| β₃ · CH | State of Charge | Battery SOC % |
| β₄ · SPD | Vehicle Speed | Speed at braking event (km/h) |
| β₅ · CUR | Current Draw | Total current draw (A) |
| ε | Error Term | Stochastic noise ~ N(0, σ²) |

---

## Statistical Validation

All 6 regression assumptions tested before model construction:

| Test | Result | Status |
|------|--------|--------|
| Shapiro-Wilk (Normality) | W = 0.8398 | ✅ PASS |
| Durbin-Watson (Independence) | DW = 2.006 | ✅ PASS |
| Homoscedasticity | Breusch-Pagan | ✅ PASS |
| VIF (Multicollinearity) | Max VIF = 1.69 | ✅ PASS |
| Linearity | Max r = 0.7743 | ✅ PASS |
| Sample Adequacy | n = 24,277 | ✅ PASS |

---

## Setup & Installation

### Prerequisites
- Python 3.11 or higher
- A modern browser (Chrome / Edge)

### Step 1 — Install dependencies

```bash
pip install numpy pandas scikit-learn scipy flask flask-cors
```

### Step 2 — Train the model

```bash
python model.py
```

This generates: `coefficients.json`, `linear_model.pkl`, `poly_model.pkl`, `poly_features.pkl`, `scaler.pkl`

### Step 3 — Start the Flask API

Open a **new terminal** and run:

```bash
python app.py
```

API will be live at: `http://localhost:5000`

### Step 4 — Open the dashboard

Double-click `index.html` in File Explorer, or open it in your browser directly.

> **Note:** The dashboard works even without the Flask API running — it loads `coefficients.json` directly for client-side predictions.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status check |
| POST | `/predict` | Predict battery stress index |
| GET | `/coefficients` | Return full model coefficients + metrics |

**Example POST request:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"BRK": 10, "CH": 60, "SPD": 57, "CUR": 28}'
```

**Example response:**

```json
{
  "battery_stress": 23.4,
  "health_status": "Moderate Stress",
  "color": "#b45309",
  "model_r2": 0.8093
}
```

---

## Dataset

**Source:** Real EV Driving Telemetry Dataset  
**Samples:** 24,277 records  
**Regen Events:** 6,624 logged regenerative braking events  

| Column | Description |
|--------|-------------|
| SPD | Vehicle speed (km/h) |
| BRK | Braking pedal force |
| CUR | Current draw (A) — negative = regen |
| CH | State of Charge (%) |
| VOL | Voltage (V) |
| ACC | Acceleration |
| ECO | Eco mode flag |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data & ML | Python, NumPy, Pandas, Scikit-learn, SciPy |
| API | Flask, Flask-CORS |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Fonts | Plus Jakarta Sans, Geist Mono |

---

## References

1. Real EV Driving Telemetry Dataset — Kaggle
2. Montgomery, D.C. — *Introduction to Linear Regression Analysis*
3. Scikit-learn Documentation — https://scikit-learn.org
4. Flask Documentation — https://flask.palletsprojects.com
5. IEEE papers on Battery Management Systems (BMS)
6. Research papers on EV battery aging and regenerative braking

---

*MathXplore 2026 · NMIT · ISE & CSE*
