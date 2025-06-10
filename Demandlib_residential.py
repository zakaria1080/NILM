import dask.dataframe as dd
import dask.array as da
import demandlib.bdew as bdew
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dask_ml.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.regularizers import l2

# Optimize TensorFlow threading for performance
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

######## Generating electricity consumption ########

# Define the reference year
year = 2020
e_slp = bdew.ElecSlp(year)

# Generate residential consumption profile
annual_demand = {"h0": 5000}  # 5000 kWh/year for a household
elec_demand = e_slp.get_profile(annual_demand).values.flatten()

# Create a time index for the entire year (hourly)
time_index = pd.date_range(start=f"{year}-01-01", periods=len(elec_demand), freq="h")

# Simulate external temperatures with noise and seasonal variations
np.random.seed(42)
days = len(time_index) // 24
temps_winter = np.linspace(-5, 10, days//4)  # Winter (Dec-Feb)
temps_spring = np.linspace(10, 20, days//4)  # Spring (Mar-May)
temps_summer = np.linspace(15, 35, days//4)  # Summer (Jun-Aug)
temps_autumn = np.linspace(20, 10, days//4)  # Autumn (Sep-Nov)
temps = np.concatenate((temps_winter, temps_spring, temps_summer, temps_autumn))
temps = np.tile(temps, (24, 1)).T.flatten()[:len(time_index)]  # Expand to a full year
temps += np.random.normal(0, 2, len(temps))  # Add Gaussian noise

# Adjust consumption based on temperature
def adjust_for_temperature(demand, temperature):
    heating_factor = np.maximum(0, 18 - temperature) * (0.02 + np.random.uniform(0, 0.005))  # Heating
    cooling_factor = np.maximum(0, temperature - 26) * (0.01 + np.random.uniform(0, 0.003))  # Air conditioning
    return demand * (1 + heating_factor + cooling_factor)

adjusted_demand = adjust_for_temperature(elec_demand, temps)

# Apply variations based on the day of the week and day/night cycles
weekday_factors = np.where(time_index.weekday < 5,
                           1.0 + np.random.uniform(-0.05, 0.05, len(time_index)),
                           1.15 + np.random.uniform(-0.05, 0.05, len(time_index)))

hourly_factors = np.where((time_index.hour >= 23) | (time_index.hour < 6),
                          0.8 + np.random.uniform(-0.1, 0.1, len(time_index)),
                          1.0 + np.random.uniform(-0.05, 0.05, len(time_index)))

random_spikes = np.random.choice([1, 1.1, 1.2, 1.5], size=len(time_index), p=[0.85, 0.1, 0.04, 0.01])

final_demand = adjusted_demand * weekday_factors * hourly_factors * random_spikes

# Create a Dask DataFrame
df = dd.from_pandas(pd.DataFrame({
    "Consumption (kW)": final_demand,
    "Temperature (°C)": temps,
    "Day of the week": time_index.weekday / 6,  # Normalize between 0 and 1
    "Hour": time_index.hour / 23  # Normalize between 0 and 1
}, index=time_index), npartitions=10)

df = df.fillna(0)  # Replace NaN values with 0

# Plot total consumption over a week
week_start = "2020-06-01"
week_end = "2020-06-07"
df_subset = df.loc[week_start:week_end].compute()
plt.figure(figsize=(14, 6))
plt.plot(df_subset.index, df_subset["Consumption (kW)"], label="Total Consumption", color="blue")
plt.xlabel("Time (Week)")
plt.ylabel("Consumption (kW)")
plt.title("Total Consumption Over a Week")
plt.legend()
plt.grid(True)
plt.show()

######## NILM Implementation with CNN + LSTM ########

# Simulating consumption of individual appliances
np.random.seed(42)

equipments = {
    "Fridge": np.random.uniform(0.1, 0.2, len(df)),
    "TV": (np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])) * np.random.uniform(0.05, 0.2, len(df)),
    "AC": (df["Temperature (°C)"].compute() > 26).astype(int) * np.random.uniform(0.5, 1.5, len(df)),
    "Washing Machine": (np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])) * np.random.uniform(0.5, 2, len(df)),
}

equipments_df = dd.from_pandas(pd.DataFrame(equipments, index=df.index), npartitions=50)

# Prepare input features and labels
X = df[["Consumption (kW)", "Temperature (°C)", "Day of the week", "Hour"]].to_dask_array(lengths=True)
Y_classification = (equipments_df.to_dask_array(lengths=True) > 0).astype(int)
Y_regression = equipments_df.to_dask_array(lengths=True)

# Normalize data
scaler_X = MinMaxScaler(feature_range=(0, 10))
scaler_Y = MinMaxScaler(feature_range=(0, 10))

X = scaler_X.fit_transform(X)
Y_regression = scaler_Y.fit_transform(Y_regression)

# Create sequences for time series modeling
def create_sequences(X, Y_class, Y_reg, seq_length=60):
    Xs, Ys_class, Ys_reg = [], [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length, :])
        Ys_class.append(Y_class[i+seq_length])
        Ys_reg.append(Y_reg[i+seq_length])
    return np.array(Xs), np.array(Ys_class), np.array(Ys_reg)

X_seq, Y_class_seq, Y_reg_seq = create_sequences(X, Y_classification, Y_regression, seq_length=60)

# Train-test split
X_train, X_test, Y_class_train, Y_class_test, Y_reg_train, Y_reg_test = train_test_split(
    X_seq, Y_class_seq, Y_reg_seq, test_size=0.2, random_state=42, shuffle=True)

# Build optimized CNN + LSTM model
input_layer = keras.Input(shape=(60, X.shape[1]))
x = layers.Conv1D(128, kernel_size=7, activation="relu", padding="same")(input_layer)
x = layers.Conv1D(256, kernel_size=5, activation="relu", padding="same")(x)
x = layers.Conv1D(512, kernel_size=3, activation="relu", padding="same")(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.LSTM(64, return_sequences=False)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.1)(x)
classification_output = layers.Dense(len(equipments), activation="sigmoid", name="classification_output")(x)
regression_output = layers.Dense(len(equipments), activation="relu", name="regression_output")(x)
model = keras.Model(inputs=input_layer, outputs=[classification_output, regression_output])

# Compilation with Huber Loss
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.005),
    loss={"classification_output": "binary_crossentropy", "regression_output": keras.losses.Huber(delta=1.0)},
    metrics={"classification_output": "accuracy", "regression_output": "mae"}
)

# Entraînement
history = model.fit(
    X_train,
    {"classification_output": Y_class_train, "regression_output": Y_reg_train},
    validation_data=(X_test, {"classification_output": Y_class_test, "regression_output": Y_reg_test}),
    epochs=50,
    batch_size=128
)


Y_pred_regression, Y_pred_classification = model.predict(X_test)

Y_pred_reg_original = scaler_Y.inverse_transform(Y_pred_regression)
Y_reg_test_original = scaler_Y.inverse_transform(Y_reg_test)

# 📌 Define a time range for the plot
time_index_test = pd.date_range(start="2020-01-01", periods=len(Y_reg_test_original), freq="H")

# Plot over a year
plt.figure(figsize=(14, 6))
plt.plot(time_index_test, Y_reg_test_original[:, 0], label="Real Consumption", linestyle="dashed", color="blue", alpha=0.6)
plt.plot(time_index_test, Y_pred_reg_original[:, 0], label="Prediction", color="red", alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Consumption (kW)")
plt.title("Comparison of consumption over an entire year (real vs predicted)")
plt.legend()
plt.grid(True)
plt.show()

# Plot over a week
week_start = "2020-06-01"
week_end = "2020-06-07"
mask = (time_index_test >= week_start) & (time_index_test <= week_end)

plt.figure(figsize=(14, 6))
plt.plot(time_index_test[mask], Y_reg_test_original[mask, 0], label="Real Consumption", linestyle="dashed", color="blue")
plt.plot(time_index_test[mask], Y_pred_reg_original[mask, 0], label="Predicted Consumption", color="red")
plt.xlabel("Time")
plt.ylabel("Consumption (kW)")
plt.title("Comparison of consumption over a week (real vs predicted)")
plt.legend()
plt.grid(True)
plt.show()

# 📌 Définir une plage de temps correspondant aux données
time_index_test = pd.date_range(start="2020-01-01", periods=len(Y_reg_test_original), freq="H")

# 📊 Graphique sur **une année complète** pour chaque équipement
equipments_names = list(equipments.keys())

plt.figure(figsize=(14, 10))
for i, equip in enumerate(equipments_names):
    plt.subplot(2, 2, i+1)  # 2 lignes, 2 colonnes
    plt.plot(time_index_test, Y_reg_test_original[:, i], label="Real Consumption", linestyle="dashed", color="blue", alpha=0.6)
    plt.plot(time_index_test, Y_pred_reg_original[:, i], label="Predicted Consumption", color="red", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Consumption (kW)")
    plt.title(f"{equip} - Prediction vs Reality (Year)")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# 📊 Graphique détaillé sur **une semaine** pour chaque équipement
week_start = "2020-06-01"
week_end = "2020-06-07"
mask = (time_index_test >= week_start) & (time_index_test <= week_end)

plt.figure(figsize=(14, 10))
for i, equip in enumerate(equipments_names):
    plt.subplot(2, 2, i+1)  # 2 lignes, 2 colonnes
    plt.plot(time_index_test[mask], Y_reg_test_original[mask, i], label="Real Consumption", linestyle="dashed", color="blue")
    plt.plot(time_index_test[mask], Y_pred_reg_original[mask, i], label="Predicted Consumption", color="red")
    plt.xlabel("Time")
    plt.ylabel("Consumption (kW)")
    plt.title(f"{equip} - Prediction vs Reality (Week)")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(history.history["classification_output_accuracy"], label="Train Accuracy")
plt.plot(history.history["val_classification_output_accuracy"], label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Classification Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(history.history["regression_output_loss"], label="Train Loss")
plt.plot(history.history["val_regression_output_loss"], label="Val Loss")
plt.xlabel("Épochs")
plt.ylabel("Loss")
plt.title("Regression Loss")
plt.legend()
plt.show()

