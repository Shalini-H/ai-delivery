# 🚚 Project Workflow: AI-Based Delivery Time Prediction in Logistics Management

### **AI Experiential Learning Event | Group 3**
- **AI Concept:** Supervised Machine Learning – Regression
- **Target Variable:** Delivery Time (Minutes / Hours)
- **Algorithm:** Random Forest Regressor
- **Tools:** Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Flask

---

## 🔄 End-to-End Architecture & Workflow

```text
+-------------------------------------------------------------+
|                     Logistics Dataset                       |
|   (Distance, Prep Time, Weight, Shipping Mode, Weather,     |
|         Traffic Level, Vehicle Type, Warehouse Hub)         |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|              Data Cleaning & Preprocessing                  |
|    - Numerical: StandardScaler (Zero Mean, Unit Variance)   |
|    - Categorical: OneHotEncoder (drop='first')              |
|    - ColumnTransformer & Pipeline Integration               |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|                Exploratory Data Analysis (EDA)              |
|    - Feature Correlation Heatmap                            |
|    - Traffic Density & Weather Impact Boxplots              |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|             Train - Test Split (80% / 20%)                  |
|    - 2,400 Training Records | 600 Testing Records           |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|          Supervised Model: Random Forest Regressor          |
|    - 100 Decision Trees Ensemble (max_depth=12)             |
|    - Captures non-linear compounding delays                 |
|    - Bootstrap aggregation for robust variance reduction    |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|               Model Evaluation & Performance                |
|    - Mean Absolute Error (MAE): 7.27 mins                   |
|    - Root Mean Squared Error (RMSE): 10.88 mins             |
|    - R² Score (Variance Explained): 92.13%                  |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|              Interactive Web Demonstration App              |
|    - Real-Time Sliders & Parameter Customization            |
|    - Instant AI Prediction & ETA Calculation                |
|    - High-Resolution Feature Importance & EDA Visuals       |
+-------------------------------------------------------------+
```

---

## 📋 Detailed Step-by-Step Workflow

### 1. Dataset Generation & Schema
* **Script:** [`generate_data.py`](generate_data.py)
* **Dataset File:** [`data/logistics_delivery_data.csv`](data/logistics_delivery_data.csv) (3,000 shipment records)
* **Features Included:**
  1. `Distance_KM`: Trip distance ranging from 1.5 km to 120 km.
  2. `Prep_Time_Min`: Order packaging and warehouse fulfillment delay (8 to 65 mins).
  3. `Package_Weight_KG`: Package weight (0.2 kg to 35 kg).
  4. `Shipping_Mode`: Service tier (`Standard`, `Express`, `Same-Day`).
  5. `Weather`: Environmental factor (`Clear`, `Rainy`, `Foggy`, `Stormy`).
  6. `Traffic_Level`: Real-time road congestion (`Low`, `Medium`, `High`, `Jam`).
  7. `Vehicle_Type`: Fleet dispatch mode (`Motorcycle`, `Delivery Van`, `Mini Truck`).
  8. `Warehouse`: Distribution origin (`Hub-Central`, `Hub-North`, `Hub-South`, `Hub-East`, `Hub-West`).
  9. `Delivery_Time_Min`: Target continuous variable (ground truth delivery duration).

---

### 2. Preprocessing & Feature Engineering
* **Script:** [`train_and_evaluate.py`](train_and_evaluate.py)
* Continuous numeric attributes (`Distance_KM`, `Prep_Time_Min`, `Package_Weight_KG`) are normalized with **`StandardScaler`**.
* Categorical attributes (`Shipping_Mode`, `Weather`, `Traffic_Level`, `Vehicle_Type`, `Warehouse`) are encoded into binary dummy vectors using **`OneHotEncoder(drop='first', handle_unknown='ignore')`** to eliminate multicollinearity.
* Packaged seamlessly inside a Scikit-learn **`ColumnTransformer`** to prevent data leakage between train and test sets.

---

### 3. Exploratory Data Analysis & Visualizations
High-resolution plots automatically saved to [`static/charts/`](static/charts/):
- **`actual_vs_predicted.png`**: Scatter plot showing actual delivery time versus Random Forest predictions ($R^2 = 0.921$).
- **`correlation_heatmap.png`**: Feature correlation matrix highlighting primary drivers of delivery time.
- **`traffic_weather_impact.png`**: Boxplots quantifying delays caused by severe traffic jams and storms.
- **`feature_importance.png`**: Ranking of top 10 features via Random Forest Gini impurity reduction.
- **`model_comparison.png`**: Visual performance scorecard across MAE, RMSE, and $R^2$.

---

### 4. Regression Model Training
Data is split into **80% training (2,400 samples)** and **20% testing (600 samples)**:

**Model: Random Forest Regressor**
- An ensemble of 100 decision trees (`n_estimators=100`, `max_depth=12`).
- Aggregates predictions across randomized bootstrapped subsets, capturing non-linear interactions (e.g., how rain compounds traffic delays).

---

### 5. Model Evaluation Results

| Performance Metric | Random Forest Regressor Result | Practical Significance |
| :--- | :---: | :--- |
| **Mean Absolute Error (MAE)** | **7.27 mins** | Predictions deviate by only ~7.2 mins on average across all shipments |
| **Root Mean Squared Error (RMSE)** | **10.88 mins** | Low dispersion with strong resilience against extreme outlier errors |
| **$R^2$ Score (Accuracy)** | **0.9213 (92.1%)** | Over 92% of the variance in logistics delivery duration is explained |

Serialized model saved in:
- [`models/random_forest_model.pkl`](models/random_forest_model.pkl)
- [`models/metrics.json`](models/metrics.json)

---

### 6. Interactive Web Demonstration
* **Script:** [`app.py`](app.py) | **Template:** [`templates/index.html`](templates/index.html)
* **Access URL:** `http://127.0.0.1:5000`
* Features:
  - Real-time sliders for distance, prep time, and weight.
  - Dropdown selectors for vehicle, weather, traffic, and hub.
  - One-click presets: *Urban Express*, *Rush-Hour Rain*, *Intercity Cargo*.
  - Instant high-accuracy Random Forest prediction and real-time arrival clock timestamp (ETA).
  - Built-in EDA gallery tab.

---

### 7. Google Colab / Notebook Workflow
* **Notebook:** [`Delivery_Time_Prediction_Colab.ipynb`](Delivery_Time_Prediction_Colab.ipynb)
* Ready to be opened in [Google Colab](https://colab.research.google.com) to present code cells, markdown explanations, and live outputs during evaluation.

---

## ⚡ Execution Commands Quick Reference

| Action | Terminal Command |
| :--- | :--- |
| **1. Generate Dataset** | `python generate_data.py` |
| **2. Train Model & Generate Plots** | `python train_and_evaluate.py` |
| **3. Run Interactive Demo App** | `python app.py` *(or double-click `run_demo.bat`)* |
| **4. Open Web Interface** | Visit `http://127.0.0.1:5000` in browser |
