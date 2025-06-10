import pandas as pd
import numpy as np
import demandlib.bdew as bdew
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.cluster import KMeans
from scipy.integrate import odeint
from sklearn.model_selection import GridSearchCV

# Load data for electric consumption
year = 2016
annual_demand = {"h0": 5000}
e_slp = bdew.ElecSlp(year)
elec_demand = e_slp.get_profile(annual_demand).values.flatten()
time_index = pd.date_range(start=f"{year}-01-01", periods=len(elec_demand), freq="h")

# Load file containing outdoor temperature
temp_df = pd.read_csv(r"C:\Users\zakar\Documents\Cours\Electromecanique\MA2\Mémoire\city_temperature.csv", low_memory=False)

# Keep Brussels, Belgium and years from 2017 to 2020
temp_df = temp_df[(temp_df["City"] == "Brussels") & (temp_df["Country"] == "Belgium")]
temp_df = temp_df[temp_df["Year"].between(2017, 2020)]

# Delete invalid temperatures
temp_df = temp_df[temp_df["AvgTemperature"] > -50]

# Create datetime column
temp_df["Date"] = pd.to_datetime(temp_df[["Year", "Month", "Day"]], errors="coerce")
temp_df = temp_df.dropna(subset=["Date"]).sort_values("Date")

# Convert temperature in Celsius
temp_df["AvgTempC"] = (temp_df["AvgTemperature"] - 32) * 5 / 9

# Linear interpolation at each hour
temp_df = temp_df.set_index("Date")
temp_hourly = temp_df["AvgTempC"].resample("h").interpolate("linear")

# Adjust length of data with elec_demand
expected_length = len(elec_demand)
if len(temp_hourly) >= expected_length:
    temp = temp_hourly.values[:expected_length]
else:
    last_val = temp_hourly.values[-1]
    temp = np.concatenate([temp_hourly.values, np.repeat(last_val, expected_length - len(temp_hourly))])


# Adjust consumption with temperature to have realistic data
def adjust_for_temperature(demand, temperature):
    heating_factor = np.maximum(0, 18 - temperature) * 0.02
    cooling_factor = np.maximum(0, temperature - 26) * 0.01
    return demand * (1 + heating_factor + cooling_factor)

adjusted_demand = adjust_for_temperature(elec_demand, temp)

# Add factors in consumption for the features
weekday_factors = np.where(time_index.weekday < 5, 1.0, 1.15)
hourly_factors = np.where((time_index.hour >= 23) | (time_index.hour < 6), 0.8, 1.0)
final_demand = adjusted_demand * weekday_factors * hourly_factors

# Create the Data Frame
df = pd.DataFrame({
    "Consumption (kW)": final_demand,
    "Temperature (°C)": temp,
    "Day of the week": time_index.weekday,
    "Hour": time_index.hour
}, index=time_index)

# Creat different periods to have a realistic consumption
def categorize_time(hour):
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 23:
        return "Evening"
    else:
        return "Night"

df["Time of Day"] = df["Hour"].apply(categorize_time)

# 🔹 Clustering (3 segments)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(df[["Consumption (kW)", "Hour"]].astype(float))

# 🔹 Features avancées
df["Temp²"] = df["Temperature (°C)"] ** 2
df["Hour²"] = df["Hour"] ** 2
df["Temp*Hour"] = df["Temperature (°C)"] * df["Hour"]
df["Lag_Consumption_1h"] = df["Consumption (kW)"].shift(1)
df["Lag_Consumption_24h"] = df["Consumption (kW)"].shift(24)
df = df.dropna()

df = pd.get_dummies(df, columns=["Time of Day"], drop_first=True)

# Add year for the split train/test
df['Year'] = df.index.year

train_years = [2016, 2017, 2018]
test_year = 2019

train_data = df[df['Year'].isin(train_years)]
test_data = df[df['Year'] == test_year]

X_train = train_data.drop(columns=["Consumption (kW)", "Cluster", "Year"])
y_train = train_data["Consumption (kW)"]
X_test = test_data.drop(columns=["Consumption (kW)", "Cluster", "Year"])
y_test = test_data["Consumption (kW)"]

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# 🔹 Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=50, min_samples_split=5, min_samples_leaf=3, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate the models
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    print(f"📊 {model_name} Performance:")
    print(f" - MAE: {mae:.3f} kW")
    print(f" - MSE: {mse:.3f}")
    print(f" - RMSE: {rmse:.3f} kW")
    print(f" - MAPE: {mape:.2f}%")
    print(f" - R²: {r2:.3f}\n")

evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")

######### Train/Test visualisation ##########

plt.figure(figsize=(14, 5))
plt.plot(train_data.index, train_data["Consumption (kW)"], label="Train data (2016–2018)", color="skyblue", linewidth=1)
plt.plot(test_data.index, test_data["Consumption (kW)"], label="Test data (2019)", color="orange", linewidth=1)
plt.xlabel("Time")
plt.ylabel("Consumption (kW)")
plt.title("Train/Test Split - Electrical Consumption")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

start_month = pd.to_datetime("2019-01-01")
end_month = pd.to_datetime("2019-02-01")

df_month = test_data.loc[start_month:end_month]
X_month = X_test.loc[start_month:end_month]
y_month = y_test.loc[start_month:end_month]

# Compute predictions for this period
y_pred_lr_month = lr_model.predict(scaler.transform(X_month))
y_pred_rf_month = rf_model.predict(scaler.transform(X_month))

rmse_lr = np.sqrt(mean_squared_error(y_month, y_pred_lr_month))
mape_lr = mean_absolute_percentage_error(y_month, y_pred_lr_month) * 100

rmse_rf = np.sqrt(mean_squared_error(y_month, y_pred_rf_month))
mape_rf = mean_absolute_percentage_error(y_month, y_pred_rf_month) * 100

plt.figure(figsize=(14, 6))
plt.plot(y_month.index, y_month.values, label="Real Consumption", linestyle="dashed", color="blue")
plt.plot(y_month.index, y_pred_lr_month, label=f"LR - RMSE: {rmse_lr:.2f}, MAPE: {mape_lr:.1f}%", color="green")
plt.plot(y_month.index, y_pred_rf_month, label=f"RF - RMSE: {rmse_rf:.2f}, MAPE: {mape_rf:.1f}%", color="red")

plt.title("Real consumption and prediction (January 2019)")
plt.xlabel("Time")
plt.ylabel("Consumption (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


###########  Comparison of the models  ##############

one_week = y_test[:24 * 7]  # 24h * 7 jours

plt.figure(figsize=(14, 6))
plt.plot(one_week.index, one_week.values, label="Actual Consumption", linestyle="dashed", color="blue")
plt.plot(one_week.index, y_pred_lr[:len(one_week)], label="Predicted LR", color="green")
plt.plot(one_week.index, y_pred_rf[:len(one_week)], label="Predicted RF", color="red")
plt.title("Prediction vs Actual Consumption (1 Week)")
plt.xlabel("Date")
plt.ylabel("Consumption (kW)")
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 1 month
one_month = y_test[:24 * 30]  # 24h * 30 jours

plt.figure(figsize=(14, 6))
plt.plot(one_month.index, one_month.values, label="Actual Consumption", linestyle="dashed", color="blue")
plt.plot(one_month.index, y_pred_lr[:len(one_month)], label="Predicted LR", color="green")
plt.plot(one_month.index, y_pred_rf[:len(one_month)], label="Predicted RF", color="red")
plt.title("Prediction vs Actual Consumption (1 Month)")
plt.xlabel("Date")
plt.ylabel("Consumption (kW)")
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


X_full = df.drop(columns=["Consumption (kW)", "Cluster", "Year"])
X_scaled = scaler.transform(X_full)

# Comparison Temperature vs Consumption
plt.figure(figsize=(12, 5))
plt.scatter(df["Temperature (°C)"], df["Consumption (kW)"], alpha=0.3, label="Actual Data", color="blue")
#plt.scatter(df["Temperature (°C)"], rf_model.predict(X_scaled), alpha=0.3, color="red", label="Predicted RF")
#plt.scatter(df["Temperature (°C)"], lr_model.predict(X_scaled), alpha=0.3, color="green", label="Predicted LR")
plt.xlabel("Temperature (°C)")
plt.ylabel("Consumption (kW)")
plt.title("Consumption Prediction Based on Temperature")
plt.legend()
plt.grid(True)
plt.show()


# Outdoor temperature plot
plt.figure(figsize=(14, 5))
plt.plot(df.index, df["Temperature (°C)"], color="orange", label="Outdoor Temperature")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Outdoor Temperature over Time")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

########### Predictions of consumption with temperature ##############

future_temperatures = np.linspace(df["Temperature (°C)"].min(), df["Temperature (°C)"].max(), 100)
df_future_temp = pd.DataFrame({
    "Temperature (°C)": future_temperatures
})

df_future_temp["Day of the week"] = 2  # Wednesday
df_future_temp["Hour"] = 12  # Midi
df_future_temp["Temp²"] = df_future_temp["Temperature (°C)"] ** 2
df_future_temp["Hour²"] = df_future_temp["Hour"] ** 2
df_future_temp["Temp*Hour"] = df_future_temp["Temperature (°C)"] * df_future_temp["Hour"]
df_future_temp["Lag_Consumption_1h"] = df["Consumption (kW)"].mean()
df_future_temp["Lag_Consumption_24h"] = df["Consumption (kW)"].mean()


df_future_temp["Time of Day_Afternoon"] = 1
df_future_temp["Time of Day_Evening"] = 0
df_future_temp["Time of Day_Morning"] = 0
df_future_temp["Time of Day_Night"] = 0

# See if all the columns are present
for col in X_full.columns:
    if col not in df_future_temp.columns:
        df_future_temp[col] = 0

df_future_temp = df_future_temp[X_full.columns]  # Réordonner les colonnes
df_future_temp_scaled = scaler.transform(df_future_temp)

# Prédictions
pred_lr_temp = lr_model.predict(df_future_temp_scaled)
pred_rf_temp = rf_model.predict(df_future_temp_scaled)

plt.figure(figsize=(12, 6))
plt.scatter(df["Temperature (°C)"], df["Consumption (kW)"], alpha=0.3, label="Données réelles", color="gray")
plt.plot(future_temperatures, pred_lr_temp, label="Prédiction LR", color="green", linewidth=2)
plt.plot(future_temperatures, pred_rf_temp, label="Prédiction RF", color="red", linewidth=2)
plt.xlabel("Température extérieure (°C)")
plt.ylabel("Consommation (kW)")
plt.title("Consommation en fonction de la température (modèles LR et RF)")
plt.legend()
plt.grid(True)
plt.show()


###### Estimation of indoor temperature and Control with PID #######

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint, dt=3600):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.dt = dt  # Pas de temps en secondes (ici 1h)
        self.integral = 0
        self.prev_error = 0

    def compute(self, current_temp):
        error = self.setpoint - current_temp
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return max(0, output)  # Chauffage ne peut pas être négatif

# Parameters of the termal model
R = 2.0  # °C/W
C = 1e5  # J/°C
Q_loss = 100.0  # pertes
T_int_initial = 17.0
T_setpoint = 20.0

# PID control
pid = PIDController(Kp=25, Ki=0.01, Kd=13000, setpoint=T_setpoint)

T_int = [T_int_initial]
Q_heating_list = []

for i in range(1, len(df)):
    T_ext_i = df["Temperature (°C)"].iloc[i]
    T_current = T_int[-1]

    Q_heating_i = pid.compute(T_current)

    # Résolution sur une heure (1 pas)
    T_next = odeint(
        lambda T, t: (T_ext_i - T) / (R * C) + (Q_heating_i - Q_loss) / C,
        T_current, [0, 3600]
    )[-1][0]

    T_int.append(T_next)
    Q_heating_list.append(Q_heating_i)

"""
# 🔹 Paramètres du modèle 2R2C
R_ia = 0.1      # Résistance entre air et mur [°C/W]
R_we = 2.0      # Résistance entre mur et extérieur [°C/W]
C_air = 1e5     # Capacité thermique de l’air intérieur [J/°C]
C_wall = 2e6    # Capacité thermique des murs [J/°C]
Q_loss = 100.0  # Pertes internes

T_air_init = 17.0
T_wall_init = 17.0
T_setpoint = 20.0

pid = PIDController(Kp=25, Ki=0.01, Kd=13000, setpoint=T_setpoint)

T_air = [T_air_init]
T_wall = [T_wall_init]
Q_heating_list = []

def model_2R2C(T, t, T_ext, Q_heat):
    T_air, T_wall = T
    dT_air = (T_wall - T_air) / (R_ia * C_air) + (Q_heat - Q_loss) / C_air
    dT_wall = (T_ext - T_wall) / (R_we * C_wall) - (T_wall - T_air) / (R_ia * C_wall)
    return [dT_air, dT_wall]

# 🔁 Simulation pas à pas (1h par boucle)
for i in range(1, len(df)):
    T_ext_i = df["Temperature (°C)"].iloc[i]
    T_air_current = T_air[-1]
    T_wall_current = T_wall[-1]

    Q_heat_i = pid.compute(T_air_current)

    sol = odeint(model_2R2C, [T_air_current, T_wall_current], [0, 3600], args=(T_ext_i, Q_heat_i))
    T_air_next, T_wall_next = sol[-1]

    T_air.append(T_air_next)
    T_wall.append(T_wall_next)
    Q_heating_list.append(Q_heat_i)

# 🔹 Ajout au DataFrame
df = df.iloc[1:].copy()
df["Indoor Temperature (°C)"] = T_air[1:]
df["Wall Temperature (°C)"] = T_wall[1:]
df["Q_heating_PID (W)"] = Q_heating_list
"""


# Add in the data frame
df = df.iloc[1:].copy()
df["Indoor Temperature (°C)"] = T_int[1:]
df["Q_heating_PID (W)"] = Q_heating_list

# Select one year
start_year = pd.Timestamp("2016-01-01")
end_year = start_year + pd.DateOffset(days=30)
df_year = df.loc[start_year:end_year]

# Visualisation
plt.figure(figsize=(16, 6))
plt.plot(df_year.index, df_year["Indoor Temperature (°C)"], label="Indoor Temp (PID)", color="red")
plt.plot(df_year.index, df_year["Temperature (°C)"], label="Outdoor Temp", linestyle="--", color="blue")
plt.axhline(T_setpoint, color="gray", linestyle="--", label="Setpoint (20°C)")
plt.title("PID Indoor Temperature Regulation - Year 2019")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 4))
plt.plot(df.index, df["Q_heating_PID (W)"], label="Heating Power", color="orange")
plt.title("Heating Power Applied by PID - Year 2019")
plt.xlabel("Time")
plt.ylabel("Power (W)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

########## Simulation of all appliances consumption and model ###########
np.random.seed(42)
n = len(df)

temp_all = df["Temperature (°C)"].values

# Heating
# Convert power in kW
df["Heating"] = df["Q_heating_PID (W)"] / 1000.0

# Dish washer: one cycle every day between 20h and 22h
df["Dishwasher"] = 0.0
for i in range(len(df) - 1):
    if df.index[i].hour == 20:
        df.iloc[i, df.columns.get_loc("Dishwasher")] = 1.2
        df.iloc[i + 1, df.columns.get_loc("Dishwasher")] = 0.9

# Fridge (cyclic + sensible temperature)
season_factor = np.where(temp_all > 20, 1.3, np.where(temp_all < 10, 0.7, 1.0))
base_fridge = np.random.normal(loc=0.15, scale=0.01, size=n)
dropout_mask = np.random.rand(n) < np.where(temp_all < 10, 0.1, 0.02)
spike_mask = np.random.rand(n) < np.where(temp_all > 20, 0.1, 0.02)
fridge = base_fridge * season_factor
fridge[dropout_mask] = 0
fridge[spike_mask] += np.random.uniform(0.2, 0.4, size=spike_mask.sum())
df["Fridge"] = fridge

# Washing Machine (1 cycle every 2 days)
washing = np.zeros(n)
for i in range(0, n, 48):
    if i + 2 < n:
        washing[i] = 0.5
        washing[i + 1] = 0.3
df["WashingMachine"] = washing

# Microwave (3 times per day)
df["Microwave"] = 0.0
for h in [8, 13, 19]:
    indices = (df.index.hour == h)
    df.loc[indices, "Microwave"] = np.random.uniform(0.1, 0.2, size=indices.sum())

# Lighting (morning and evening + more in winter)
def lamp_usage(hour, month):
    if 6 <= hour < 8 or 18 <= hour <= 22:
        seasonal_factor = 1.0 if month in [12, 1, 2] else 0.6
        return np.random.uniform(0.1, 0.3) * seasonal_factor
    return 0.0
df["Lighting"] = [lamp_usage(h, m) for h, m in zip(df.index.hour, df.index.month)]

# Game console during the evenings and the week-ends
df["GameConsole"] = 0.0
for idx in df.index:
    hour = idx.hour
    day = idx.dayofweek  # 0 = lundi, 6 = dimanche
    if (17 <= hour <= 23 and np.random.rand() < 0.15) or (day >= 5 and 14 <= hour <= 23 and np.random.rand() < 0.3):
        df.at[idx, "GameConsole"] = np.random.uniform(0.12, 0.2)

# TV: morning (6–8h) and evening (18–23h)
df["TV"] = 0.0
for idx in df.index:
    hour = idx.hour
    if 6 <= hour < 8 and np.random.rand() < 0.3:
        df.at[idx, "TV"] = np.random.uniform(0.08, 0.12)
    elif 18 <= hour <= 23 and np.random.rand() < 0.6:
        df.at[idx, "TV"] = np.random.uniform(0.1, 0.15)

# Water Heater
water_heater = np.zeros(n)

for i, (hour, month) in enumerate(zip(df.index.hour, df.index.month)):
    seasonal_factor = 1.0 if month in [12, 1, 2] else 0.7
    morning_use = (6 <= hour < 8) and (np.random.rand() < 0.5 * seasonal_factor)
    evening_use = (18 <= hour < 20) and (np.random.rand() < 0.6 * seasonal_factor)

    if morning_use or evening_use:
        water_heater[i] = np.random.uniform(2.0, 3.0)

df["WaterHeater"] = water_heater

train_mask = df.index.year < 2019
test_mask = df.index.year == 2019

df["Hour"] = df.index.hour
df["Day of the week"] = df.index.dayofweek


def safe_mape(y_true, y_pred, threshold=0.01):
    mask = y_true > threshold
    if np.any(mask):
        return mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
    else:
        return float('nan')

# Error evaluation
def evaluate(true, pred, label=""):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mape = safe_mape(true, pred)
    r2 = r2_score(true, pred)

    print(f"📊 {label} Errors:")
    print(f" - MAE  : {mae:.4f} kW")
    print(f" - RMSE : {rmse:.4f} kW")
    print(f" - MAPE : {mape:.2f} %")
    print(f" - R²   : {r2:.4f}\n")


def get_mape(y_true, y_pred, threshold=0.05):
    """MAPE en ignorant les valeurs trop faibles de y_true"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true > threshold
    if np.sum(mask) == 0:
        return float("nan")
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# Plot for 1 week, 1 month, 3 months for each device

def plot_appliance_comparison_with_mape_in_legend(appliance, real, predicted, index, label):
    mape_val = get_mape(real[:len(index)], predicted[:len(index)])
    plt.figure(figsize=(14, 4))
    plt.plot(index, real[:len(index)], label=f"Real {label}")
    plt.plot(index, predicted[:len(index)], label=f"Predicted {label} (MAPE={mape_val:.2f}%)")
    plt.title(f"{label} - Real vs Predicted ({len(index)//24} Day Sample)")
    plt.xlabel("Time")
    plt.ylabel("Consumption (kW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_appliance_with_temperature(appliance, real, predicted, index, label, df):
    temp_series = df.loc[index, "Temperature (°C)"]

    fig, ax1 = plt.subplots(figsize=(14, 4))

    ax1.plot(index, real[:len(index)], label=f"Real {label}", color="blue", linestyle="--")
    ax1.plot(index, predicted[:len(index)], label=f"Predicted {label}", color="green")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Consumption (kW)", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(index, temp_series[:len(index)], label="Outdoor Temp", color="tab:red", linestyle="dotted", alpha=0.6)
    ax2.set_ylabel("Temperature (°C)", color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    fig.suptitle(f"{label} - Real vs Predicted + Temperature ({len(index) // 24} Day Sample)")
    fig.legend(loc="upper right")
    fig.tight_layout()
    plt.grid(True)
    plt.show()


df["Total_Consumption"] = df[["Fridge", "WashingMachine", "Microwave", "Lighting", "Heating", "Dishwasher", "GameConsole", "TV", "WaterHeater"]].sum(axis=1)

# Define feature for disaggrigation
features_disagg = df[["Total_Consumption", "Temperature (°C)", "Indoor Temperature (°C)", "Hour", "Day of the week"]]

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(features_disagg)

# Targets for each appliance
targets = {
    "Fridge": df["Fridge"],
    "WashingMachine": df["WashingMachine"],
    "Microwave": df["Microwave"],
    "Lighting": df["Lighting"],
    "Heating": df["Heating"],
    "Dishwasher": df["Dishwasher"],
    "GameConsole": df["GameConsole"],
    "TV": df["TV"],
    "WaterHeater": df["WaterHeater"]
}

scaled_targets = {}
inverse_scalers = {}

for appliance, target in targets.items():
    scaler_y = MinMaxScaler()
    scaled_targets[appliance] = scaler_y.fit_transform(target.values.reshape(-1, 1))
    inverse_scalers[appliance] = scaler_y


train_mask = df.index.year < 2019
test_mask = df.index.year == 2019
X_train = X_scaled[train_mask]
X_test = X_scaled[test_mask]

predictions = {}
true_values = {}

for appliance in targets:
    y_train = scaled_targets[appliance][train_mask]
    y_test = scaled_targets[appliance][test_mask]

    model = RandomForestRegressor(n_estimators=100, max_depth=30, random_state=42)
    model.fit(X_train, y_train.ravel())

    y_pred = model.predict(X_test)
    y_pred_inv = inverse_scalers[appliance].inverse_transform(y_pred.reshape(-1, 1))

    predictions[appliance] = y_pred_inv.flatten()
    true_values[appliance] = df.loc[test_mask, appliance]


for appliance in predictions:
    evaluate(true_values[appliance], predictions[appliance], label=appliance)


predicted_total = sum(predictions.values())
true_total = df.loc[test_mask, "Total_Consumption"]
evaluate(true_total, predicted_total, label="Total (All Appliances)")

sample_index_1w = df.loc[test_mask].index[:24 * 7]
sample_index_1m = df.loc[test_mask].index[:24 * 30]
sample_index_3m = df.loc[test_mask].index[:24 * 90]

# Plot for each device
for appliance in predictions:
    real = true_values[appliance]
    pred = predictions[appliance]

    plot_appliance_comparison_with_mape_in_legend(appliance, real, pred, sample_index_1w, appliance)
    plot_appliance_comparison_with_mape_in_legend(appliance, real, pred, sample_index_1m, appliance)
    plot_appliance_comparison_with_mape_in_legend(appliance, real, pred, sample_index_3m, appliance)

    plot_appliance_with_temperature(appliance, real, pred, sample_index_1w, appliance, df)
    plot_appliance_with_temperature(appliance, real, pred, sample_index_1m, appliance, df)
    plot_appliance_with_temperature(appliance, real, pred, sample_index_3m, appliance, df)

# Plot of the sum of all consumptions
total_mape = get_mape(true_total, predicted_total)
plt.figure(figsize=(14, 6))
plt.plot(true_total.index, true_total.values, label="Total Real", linestyle="dashed", color="black")
for appliance, pred in predictions.items():
    mape_appliance = get_mape(true_values[appliance], pred)
    plt.plot(true_total.index, pred, label=f"Predicted {appliance} (MAPE={mape_appliance:.2f}%)")
plt.plot(true_total.index, predicted_total, label=f"Sum of Predictions (MAPE={total_mape:.2f}%)", color="red", linestyle=":")
plt.title("Energy Disaggregation (2019) - All Appliances")
plt.xlabel("Time")
plt.ylabel("Consumption (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

##### GridSearch ####

# # Example with Fridge
# y_train_fridge = scaled_targets["Fridge"][train_mask]
# param_grid = {
#     "n_estimators": [50, 100],
#     "max_depth": [10, 30, 50, 100, None],
#     "min_samples_split": [2, 5],
#     "min_samples_leaf": [1, 3]
# }
#
# grid_search = GridSearchCV(
#     RandomForestRegressor(random_state=42),
#     param_grid,
#     cv=3,
#     scoring="neg_mean_absolute_error",
#     n_jobs=-1,
#     verbose=1
# )
#
# grid_search.fit(X_train, y_train_fridge.ravel())
# print("Best RF Params:", grid_search.best_params_)

