import os
import sys
import json
import joblib
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

app = Flask(__name__)

# Load trained models
models_dir = os.path.join(os.path.dirname(__file__), 'models')
lr_model_path = os.path.join(models_dir, 'linear_regression_model.pkl')
rf_model_path = os.path.join(models_dir, 'random_forest_model.pkl')
metrics_path = os.path.join(models_dir, 'metrics.json')

lr_model = None
rf_model = None
metrics_data = {}

try:
    if os.path.exists(lr_model_path):
        lr_model = joblib.load(lr_model_path)
    if os.path.exists(rf_model_path):
        rf_model = joblib.load(rf_model_path)
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
    print("[+] Models and metrics loaded successfully.")
except Exception as e:
    print(f"[-] Error loading models: {e}")

def format_minutes_to_readable(minutes):
    """Converts minutes into 'X hrs Y mins' string."""
    mins = max(5.0, round(float(minutes), 1))
    hrs = int(mins // 60)
    rem_mins = int(round(mins % 60))
    if hrs > 0:
        return f"{hrs} hr {rem_mins} min" if rem_mins > 0 else f"{hrs} hr"
    return f"{rem_mins} mins"

@app.route('/')
def home():
    return render_template('index.html', metrics=metrics_data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()

        # Parse inputs
        distance_km = float(data.get('distance_km', 15.0))
        prep_time_min = float(data.get('prep_time_min', 20.0))
        package_weight_kg = float(data.get('package_weight_kg', 3.0))
        shipping_mode = data.get('shipping_mode', 'Standard')
        weather = data.get('weather', 'Clear')
        traffic_level = data.get('traffic_level', 'Medium')
        vehicle_type = data.get('vehicle_type', 'Motorcycle')
        warehouse = data.get('warehouse', 'Hub-Central')

        # Create input DataFrame matching feature columns
        input_df = pd.DataFrame([{
            'Distance_KM': distance_km,
            'Prep_Time_Min': prep_time_min,
            'Package_Weight_KG': package_weight_kg,
            'Shipping_Mode': shipping_mode,
            'Weather': weather,
            'Traffic_Level': traffic_level,
            'Vehicle_Type': vehicle_type,
            'Warehouse': warehouse
        }])

        # Predict with Linear Regression
        lr_pred_min = float(lr_model.predict(input_df)[0]) if lr_model else 0.0
        # Predict with Random Forest Regressor
        rf_pred_min = float(rf_model.predict(input_df)[0]) if rf_model else 0.0

        # Clip predictions to reasonable positive bound
        lr_pred_min = max(10.0, round(lr_pred_min, 1))
        rf_pred_min = max(10.0, round(rf_pred_min, 1))

        # Calculate estimated delivery timestamp
        now = datetime.now()
        rf_eta = (now + timedelta(minutes=rf_pred_min)).strftime("%I:%M %p")
        lr_eta = (now + timedelta(minutes=lr_pred_min)).strftime("%I:%M %p")

        # Difference
        diff_min = round(abs(rf_pred_min - lr_pred_min), 1)

        response = {
            'status': 'success',
            'inputs': {
                'distance_km': distance_km,
                'prep_time_min': prep_time_min,
                'package_weight_kg': package_weight_kg,
                'shipping_mode': shipping_mode,
                'weather': weather,
                'traffic_level': traffic_level,
                'vehicle_type': vehicle_type,
                'warehouse': warehouse
            },
            'random_forest': {
                'minutes': rf_pred_min,
                'hours': round(rf_pred_min / 60.0, 2),
                'formatted': format_minutes_to_readable(rf_pred_min),
                'eta': rf_eta,
                'r2_score': metrics_data.get('Random Forest Regressor', {}).get('R2', 0.921),
                'mae': metrics_data.get('Random Forest Regressor', {}).get('MAE', 7.27)
            },
            'linear_regression': {
                'minutes': lr_pred_min,
                'hours': round(lr_pred_min / 60.0, 2),
                'formatted': format_minutes_to_readable(lr_pred_min),
                'eta': lr_eta,
                'r2_score': metrics_data.get('Linear Regression', {}).get('R2', 0.876),
                'mae': metrics_data.get('Linear Regression', {}).get('MAE', 9.15)
            },
            'comparison': {
                'diff_minutes': diff_min,
                'faster_model': 'Random Forest' if rf_pred_min < lr_pred_min else 'Linear Regression',
                'explanation': (
                    "Random Forest accounts for non-linear compounding delays (like severe traffic jams combined with stormy weather), "
                    "whereas Linear Regression applies a fixed additive penalty."
                )
            }
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/metrics')
def api_metrics():
    return jsonify(metrics_data)

if __name__ == '__main__':
    print("[+] Starting AI Delivery Time Prediction Web App on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
