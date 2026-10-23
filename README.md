# 🚚 AI-Based Delivery Time Prediction in Logistics Management

### **AI Experiential Learning Event | Group 3**
**AI Concept Implemented:** Supervised Machine Learning – Regression  
**Tools Used:** Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Flask  

---

## 📌 Project Overview
Predicting accurate delivery time is essential for logistics operations, route planning, customer satisfaction, and warehouse management. This project applies **Supervised Machine Learning Regression** models to predict shipment delivery times (in minutes and hours) based on dynamic operational parameters:
- **Delivery Distance** (km)
- **Preparation & Dispatch Delay** (mins)
- **Package Weight** (kg)
- **Shipping Mode** (Standard, Express, Same-Day)
- **Traffic Congestion Level** (Low, Medium, High, Jam)
- **Weather Condition** (Clear, Rainy, Foggy, Stormy)
- **Vehicle Mode** (Motorcycle, Delivery Van, Mini Truck)
- **Warehouse Location Hub**

---

## 🤖 Machine Learning Models Implemented & Evaluated

Two regression models were trained, evaluated, and compared on a 3,000-order logistics dataset (80% train, 20% test split):

| Algorithm | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | R² Score | Performance & Insights |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** (Baseline) | **9.15 mins** | 13.64 mins | **0.8761 (87.6%)** | Fits linear trend line; simple and fast. |
| **Random Forest Regressor** (Best Model) | **7.27 mins** | **10.88 mins** | **0.9213 (92.1%)** | **20.5% lower error!** Captures non-linear compounding delays (e.g. storms + traffic jams). |

---

## 📊 Visualizations Generated (`static/charts/`)
1. **`actual_vs_predicted.png`**: Scatter plot comparing predicted vs actual delivery durations.
2. **`correlation_heatmap.png`**: Heatmap showing feature correlations with delivery time.
3. **`traffic_weather_impact.png`**: Boxplots analyzing impact of traffic jams and rain/storms.
4. **`feature_importance.png`**: Top features ranking derived from Random Forest.
5. **`model_comparison.png`**: Side-by-side comparison bar charts for MAE, RMSE, and R².

---

## 🚀 How to Run the Working Demonstration

### 1. Run the Interactive Web Demonstration:
```bash
python app.py
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

### What you can demonstrate to your teacher:
1. **Interactive Control Sliders:** Adjust distance, weather, traffic, and vehicle type.
2. **One-Click Test Scenarios:**
   - 🏍️ *Urban Express* (5 km, low traffic, clear weather)
   - 🌧️ *Rush-Hour Rain* (22 km, high traffic, storm)
   - 🚛 *Intercity Cargo* (68 km, traffic jam, heavy truck)
3. **Side-by-Side Model Predictions:** Shows Random Forest vs Linear Regression predictions with exact ETA.
4. **Interactive Chart Gallery:** View all 5 high-resolution evaluation plots right on screen.
5. **Embedded Viva Guide:** Quick answers to viva voce questions.

---

## 📓 How to Run in Google Colab / Jupyter Notebook
If your teacher asks to see the notebook code:
1. Open Google Colab: [https://colab.research.google.com](https://colab.research.google.com)
2. Click **File -> Upload Notebook** and select `Delivery_Time_Prediction_Colab.ipynb` from this folder.
3. Upload `data/logistics_delivery_data.csv` to the session files.
4. Click **Runtime -> Run All**!

---

## 🎓 Teacher / Evaluator Viva Q&A Cheat Sheet

- **Q1: Why use Regression instead of Classification?**
  - *Answer:* Delivery duration is a continuous quantitative value (minutes/hours), not a discrete category. Regression models predict exact continuous time.
- **Q2: Why does Random Forest outperform Linear Regression?**
  - *Answer:* Delivery delays in the physical world are non-linear. Severe traffic combined with heavy rain creates exponential delays that a linear hyperplane cannot model. Random forest decision trees capture feature interactions effectively.
- **Q3: What do the evaluation metrics represent?**
  - *Answer:* **MAE (7.27 mins)** shows average prediction error. **RMSE (10.88 mins)** penalizes extreme delays/outliers. **R² (92.1%)** proves our model accounts for over 92% of the variance in delivery times.
