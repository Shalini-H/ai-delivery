# 🚚 AI-Based Delivery Time Prediction in Logistics Management

### **AI Experiential Learning Event | Group 3**
**AI Concept Implemented:** Supervised Machine Learning – Regression  
**Algorithm:** Random Forest Regressor  
**Tools Used:** Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Flask  

---

## 📌 Project Overview
Predicting accurate delivery time is essential for logistics operations, route planning, customer satisfaction, and warehouse management. This project applies a **Supervised Random Forest Regression** model to predict shipment delivery times (in minutes and hours) based on dynamic operational parameters:
- **Delivery Distance** (km)
- **Preparation & Dispatch Delay** (mins)
- **Package Weight** (kg)
- **Shipping Mode** (Standard, Express, Same-Day)
- **Traffic Congestion Level** (Low, Medium, High, Jam)
- **Weather Condition** (Clear, Rainy, Foggy, Stormy)
- **Vehicle Mode** (Motorcycle, Delivery Van, Mini Truck)
- **Warehouse Location Hub**

---

## 🤖 Machine Learning Model & Performance Scorecard

The **Random Forest Regressor** (100 decision trees) was trained and evaluated on a 3,000-order logistics dataset (80% train, 20% test split):

| Algorithm | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | R² Score | Performance & Insights |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest Regressor** | **7.27 mins** | **10.88 mins** | **0.9213 (92.1%)** | High-precision non-linear regression capturing compound traffic & weather delays. |

---

## 📊 Visualizations Generated (`static/charts/`)
1. **`actual_vs_predicted.png`**: Scatter plot showing actual delivery time versus Random Forest predictions ($R^2 = 0.921$).
2. **`correlation_heatmap.png`**: Heatmap showing feature correlations with delivery time.
3. **`traffic_weather_impact.png`**: Boxplots analyzing impact of traffic jams and rain/storms.
4. **`feature_importance.png`**: Top features ranking derived from Random Forest Gini impurity reduction.
5. **`model_comparison.png`**: Performance metrics scorecard (MAE, RMSE, $R^2$).

---

## 🚀 How to Run the Working Demonstration

### 1. Run the Interactive Web Demonstration:
```bash
python app.py
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

### What you can demonstrate:
1. **Interactive Control Sliders:** Adjust distance, weather, traffic, prep time, and vehicle type.
2. **One-Click Test Scenarios:**
   - 🏍️ *Urban Express* (5 km, low traffic, clear weather)
   - 🌧️ *Rush-Hour Rain* (22 km, high traffic, storm)
   - 🚛 *Intercity Cargo* (68 km, traffic jam, heavy truck)
3. **Random Forest Prediction:** Instant delivery duration (mins & hrs) with real-time arrival clock timestamp (ETA).
4. **Interactive Chart Gallery:** View all 5 high-resolution evaluation plots right on screen.

---

## 📓 How to Run in Google Colab / Jupyter Notebook
If you want to run the notebook:
1. Open Google Colab: [https://colab.research.google.com](https://colab.research.google.com)
2. Click **File -> Upload Notebook** and select `Delivery_Time_Prediction_Colab.ipynb`.
3. Upload `data/logistics_delivery_data.csv` to the session files.
4. Click **Runtime -> Run All**!
