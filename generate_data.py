import numpy as np
import pandas as pd
import os
import sys

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def generate_logistics_dataset(n_samples=3000, random_state=42):
    """
    Generates a realistic logistics delivery dataset with realistic relationships
    between distance, traffic, weather, shipping mode, vehicle type, and delivery time.
    """
    np.random.seed(random_state)

    # 1. Distance in kilometers (1 km to 120 km)
    # Using log-normal / gamma distribution to represent city + intercity orders
    distance_km = np.round(np.random.gamma(shape=2.5, scale=8.0, size=n_samples) + 1.0, 1)
    distance_km = np.clip(distance_km, 1.5, 120.0)

    # 2. Shipping Mode
    shipping_modes = ['Standard', 'Express', 'Same-Day']
    shipping_mode_probs = [0.55, 0.30, 0.15]
    shipping_mode = np.random.choice(shipping_modes, size=n_samples, p=shipping_mode_probs)

    # 3. Weather Conditions
    weather_conditions = ['Clear', 'Rainy', 'Foggy', 'Stormy']
    weather_probs = [0.65, 0.20, 0.10, 0.05]
    weather = np.random.choice(weather_conditions, size=n_samples, p=weather_probs)

    # 4. Traffic Density
    traffic_levels = ['Low', 'Medium', 'High', 'Jam']
    traffic_probs = [0.25, 0.40, 0.25, 0.10]
    traffic = np.random.choice(traffic_levels, size=n_samples, p=traffic_probs)

    # 5. Vehicle Type
    vehicle_types = ['Motorcycle', 'Delivery Van', 'Mini Truck']
    vehicle_probs = [0.45, 0.40, 0.15]
    vehicle = np.random.choice(vehicle_types, size=n_samples, p=vehicle_probs)

    # 6. Warehouse / Hub Location
    warehouses = ['Hub-North', 'Hub-South', 'Hub-Central', 'Hub-East', 'Hub-West']
    warehouse_loc = np.random.choice(warehouses, size=n_samples)

    # 7. Order Preparation / Dispatch Delay (in minutes, 10 to 60 mins)
    prep_time_min = np.round(np.random.normal(loc=25, scale=8, size=n_samples), 0)
    prep_time_min = np.clip(prep_time_min, 8, 65)

    # 8. Package Weight (kg)
    package_weight_kg = np.round(np.random.exponential(scale=3.5, size=n_samples) + 0.5, 2)
    package_weight_kg = np.clip(package_weight_kg, 0.2, 35.0)

    # Compute realistic delivery time (in minutes) based on physics & operational factors:
    # Base speed by vehicle (km/h)
    speed_map = {'Motorcycle': 38.0, 'Delivery Van': 32.0, 'Mini Truck': 26.0}
    
    # Traffic speed multiplier
    traffic_factor = {'Low': 1.15, 'Medium': 1.0, 'High': 0.70, 'Jam': 0.45}

    # Weather speed multiplier
    weather_factor = {'Clear': 1.0, 'Foggy': 0.85, 'Rainy': 0.75, 'Stormy': 0.55}

    # Shipping mode priority factor (Express gets prioritized dispatch & route)
    mode_speedup = {'Standard': 1.0, 'Express': 0.85, 'Same-Day': 0.75}

    transit_time_min = []
    for i in range(n_samples):
        base_spd = speed_map[vehicle[i]]
        tr_mult = traffic_factor[traffic[i]]
        we_mult = weather_factor[weather[i]]
        effective_speed = base_spd * tr_mult * we_mult  # km/h
        
        # Transit time in minutes = (distance / speed) * 60
        t_transit = (distance_km[i] / effective_speed) * 60.0
        
        # Priority shipping reduces prep and dispatch overhead
        t_total = (prep_time_min[i] * mode_speedup[shipping_mode[i]]) + t_transit
        
        # Add random operational noise (stoplights, customer handoff, elevator delays)
        noise = np.random.normal(0, 4.5)
        t_final = max(15.0, t_total + noise)
        transit_time_min.append(np.round(t_final, 1))

    # Delivery time in hours (for easy reading alongside minutes)
    delivery_time_hours = np.round(np.array(transit_time_min) / 60.0, 2)

    df = pd.DataFrame({
        'Distance_KM': distance_km,
        'Prep_Time_Min': prep_time_min,
        'Package_Weight_KG': package_weight_kg,
        'Shipping_Mode': shipping_mode,
        'Weather': weather,
        'Traffic_Level': traffic,
        'Vehicle_Type': vehicle,
        'Warehouse': warehouse_loc,
        'Delivery_Time_Min': transit_time_min,
        'Delivery_Time_Hours': delivery_time_hours
    })

    os.makedirs('data', exist_ok=True)
    csv_path = os.path.join('data', 'logistics_delivery_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"[+] Generated dataset with {n_samples} records saved to {csv_path}")
    print(df.head())
    print("\nDataset Summary:")
    print(df.describe())
    return df

if __name__ == '__main__':
    generate_logistics_dataset()
