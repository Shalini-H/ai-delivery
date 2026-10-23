import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for server/script plotting
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def train_and_evaluate():
    print("=" * 65, flush=True)
    print("AI-BASED DELIVERY TIME PREDICTION - MODEL TRAINING & EVALUATION", flush=True)
    print("=" * 65, flush=True)

    data_path = os.path.join('data', 'logistics_delivery_data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Run generate_data.py first.")

    df = pd.read_csv(data_path)
    print(f"[+] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns", flush=True)

    # Define features and target
    feature_cols = [
        'Distance_KM',
        'Prep_Time_Min',
        'Package_Weight_KG',
        'Shipping_Mode',
        'Weather',
        'Traffic_Level',
        'Vehicle_Type',
        'Warehouse'
    ]
    target_col = 'Delivery_Time_Min'

    X = df[feature_cols]
    y = df[target_col]

    # Preprocessing Pipeline
    num_features = ['Distance_KM', 'Prep_Time_Min', 'Package_Weight_KG']
    cat_features = ['Shipping_Mode', 'Weather', 'Traffic_Level', 'Vehicle_Type', 'Warehouse']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
        ]
    )

    # Train / Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"[+] Train set size: {X_train.shape[0]} samples", flush=True)
    print(f"[+] Test set size:  {X_test.shape[0]} samples", flush=True)

    # 1. Model 1: Linear Regression
    print("\n--- Training Model 1: Linear Regression ---", flush=True)
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)

    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_rmse = root_mean_squared_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)

    print(f"  Linear Regression MAE:  {lr_mae:.2f} mins")
    print(f"  Linear Regression RMSE: {lr_rmse:.2f} mins")
    print(f"  Linear Regression R2:   {lr_r2 * 100:.2f}%")

    # 2. Model 2: Random Forest Regression
    print("\n--- Training Model 2: Random Forest Regressor ---", flush=True)
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=1))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_pred = rf_pipeline.predict(X_test)

    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = root_mean_squared_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    print(f"  Random Forest MAE:  {rf_mae:.2f} mins")
    print(f"  Random Forest RMSE: {rf_rmse:.2f} mins")
    print(f"  Random Forest R2:   {rf_r2 * 100:.2f}%")

    # Comparison Table
    print("\n" + "=" * 65)
    print(f"{'Model Name':<28} | {'MAE (mins)':<12} | {'RMSE (mins)':<12} | {'R2 Score':<10}")
    print("-" * 65)
    print(f"{'Linear Regression':<28} | {lr_mae:<12.2f} | {lr_rmse:<12.2f} | {lr_r2:<10.4f}")
    print(f"{'Random Forest Regressor':<28} | {rf_mae:<12.2f} | {rf_rmse:<12.2f} | {rf_r2:<10.4f}")
    print("=" * 65)

    # Save Models and Artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(lr_pipeline, os.path.join('models', 'linear_regression_model.pkl'))
    joblib.dump(rf_pipeline, os.path.join('models', 'random_forest_model.pkl'))
    print("\n[+] Trained models saved successfully in /models")

    # Save metrics JSON for the demo UI
    metrics = {
        'Linear Regression': {
            'MAE': round(float(lr_mae), 2),
            'RMSE': round(float(lr_rmse), 2),
            'R2': round(float(lr_r2), 4),
            'Accuracy_Pct': round(float(lr_r2 * 100), 1)
        },
        'Random Forest Regressor': {
            'MAE': round(float(rf_mae), 2),
            'RMSE': round(float(rf_rmse), 2),
            'R2': round(float(rf_r2), 4),
            'Accuracy_Pct': round(float(rf_r2 * 100), 1)
        },
        'Total_Samples': len(df),
        'Train_Samples': len(X_train),
        'Test_Samples': len(X_test)
    }
    with open(os.path.join('models', 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # Generate Evaluation & EDA Plots
    os.makedirs(os.path.join('static', 'charts'), exist_ok=True)
    generate_plots(df, y_test, lr_pred, rf_pred, rf_pipeline, preprocessor, num_features, cat_features)

    return metrics

def generate_plots(df, y_test, lr_pred, rf_pred, rf_pipeline, preprocessor, num_features, cat_features):
    print("\n[+] Generating high-resolution presentation charts...")
    sns.set_theme(style="whitegrid", palette="muted")

    # Plot 1: Actual vs Predicted for Both Models
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Linear Regression scatter
    axes[0].scatter(y_test, lr_pred, alpha=0.45, color='#3b82f6', edgecolor='none', s=25)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_title(f'Linear Regression (R² = {r2_score(y_test, lr_pred):.3f})', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Actual Delivery Time (minutes)')
    axes[0].set_ylabel('Predicted Delivery Time (minutes)')
    axes[0].legend()

    # Random Forest scatter
    axes[1].scatter(y_test, rf_pred, alpha=0.45, color='#10b981', edgecolor='none', s=25)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_title(f'Random Forest Regressor (R² = {r2_score(y_test, rf_pred):.3f})', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Actual Delivery Time (minutes)')
    axes[1].set_ylabel('Predicted Delivery Time (minutes)')
    axes[1].legend()

    plt.suptitle('Model Evaluation: Actual vs. Predicted Delivery Time', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'charts', 'actual_vs_predicted.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Plot 2: Correlation Heatmap for Numerical Features
    plt.figure(figsize=(8, 6))
    num_df = df[['Distance_KM', 'Prep_Time_Min', 'Package_Weight_KG', 'Delivery_Time_Min']]
    corr = num_df.corr()
    sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f', square=True, linewidths=1.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap: Logistics Attributes vs Delivery Time', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'charts', 'correlation_heatmap.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Plot 3: Traffic Level & Weather Impact on Delivery Time
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(ax=axes[0], data=df, x='Traffic_Level', y='Delivery_Time_Min', palette='Blues_r', order=['Low', 'Medium', 'High', 'Jam'])
    axes[0].set_title('Impact of Traffic Congestion on Delivery Time', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Delivery Time (minutes)')
    axes[0].set_xlabel('Traffic Density')

    sns.boxplot(ax=axes[1], data=df, x='Weather', y='Delivery_Time_Min', palette='Purples_r', order=['Clear', 'Rainy', 'Foggy', 'Stormy'])
    axes[1].set_title('Impact of Weather Conditions on Delivery Time', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Delivery Time (minutes)')
    axes[1].set_xlabel('Weather Condition')

    plt.tight_layout()
    plt.savefig(os.path.join('static', 'charts', 'traffic_weather_impact.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Plot 4: Feature Importance (from Random Forest)
    try:
        rf_regressor = rf_pipeline.named_steps['regressor']
        fitted_prep = rf_pipeline.named_steps['preprocessor']
        cat_encoder = fitted_prep.named_transformers_['cat']
        encoded_cat_names = cat_encoder.get_feature_names_out(cat_features).tolist()
        all_feature_names = num_features + encoded_cat_names

        importances = rf_regressor.feature_importances_
        fi_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10)

        plt.figure(figsize=(10, 5))
        sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis')
        plt.title('Top 10 Feature Importances (Random Forest Model)', fontsize=13, fontweight='bold')
        plt.xlabel('Relative Importance Score')
        plt.tight_layout()
        plt.savefig(os.path.join('static', 'charts', 'feature_importance.png'), dpi=200, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[-] Could not plot feature importances: {e}")

    # Plot 5: Model Comparison Bar Chart (MAE, RMSE, R2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    models = ['Linear Reg', 'Random Forest']
    
    # MAE
    lr_m = mean_absolute_error(y_test, lr_pred)
    rf_m = mean_absolute_error(y_test, rf_pred)
    axes[0].bar(models, [lr_m, rf_m], color=['#3b82f6', '#10b981'], width=0.5)
    axes[0].set_title('Mean Absolute Error (MAE)\n(Lower is Better)', fontweight='bold')
    axes[0].set_ylabel('Minutes')
    for i, v in enumerate([lr_m, rf_m]):
        axes[0].text(i, v + 0.5, f"{v:.2f} m", ha='center', fontweight='bold')

    # RMSE
    lr_r = root_mean_squared_error(y_test, lr_pred)
    rf_r = root_mean_squared_error(y_test, rf_pred)
    axes[1].bar(models, [lr_r, rf_r], color=['#3b82f6', '#10b981'], width=0.5)
    axes[1].set_title('Root Mean Squared Error (RMSE)\n(Lower is Better)', fontweight='bold')
    axes[1].set_ylabel('Minutes')
    for i, v in enumerate([lr_r, rf_r]):
        axes[1].text(i, v + 0.5, f"{v:.2f} m", ha='center', fontweight='bold')

    # R2 Score
    lr_score = r2_score(y_test, lr_pred)
    rf_score = r2_score(y_test, rf_pred)
    axes[2].bar(models, [lr_score, rf_score], color=['#3b82f6', '#10b981'], width=0.5)
    axes[2].set_title('R² Score (Accuracy fit)\n(Higher is Better)', fontweight='bold')
    axes[2].set_ylabel('Score (0 to 1)')
    axes[2].set_ylim(0, 1.1)
    for i, v in enumerate([lr_score, rf_score]):
        axes[2].text(i, v + 0.03, f"{v:.3f} ({v*100:.1f}%)", ha='center', fontweight='bold')

    plt.suptitle('Performance Metric Comparison: Linear Regression vs Random Forest', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'charts', 'model_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print("[+] All presentation charts generated and saved in static/charts/")

if __name__ == '__main__':
    train_and_evaluate()
