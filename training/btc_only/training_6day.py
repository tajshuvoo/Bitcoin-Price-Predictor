import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, Dropout, Add
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# ✅ CHECK GPU AVAILABILITY
# -------------------------------
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available:", gpus)
    tf.debugging.set_log_device_placement(False)
else:
    print("GPU not available. Using CPU.")

# -------------------------------
# 📄 LOAD DATA FROM CSV
# -------------------------------
data_path = "data/raw/btc_price_1y.csv"
btc_data = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')

# -------------------------------
# 🔢 LOAD EXISTING SCALER
# -------------------------------
features = [
    "open", "high", "low", "close", "volume",
    "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume"
]
feature_data = btc_data[features]

scaler_path = "model/btc_only/btc_scaler_1day.pkl"
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

feature_data_scaled = scaler.transform(feature_data)

# -------------------------------
# 🔁 CREATE SEQUENCES (15min * 192 = 2 days)
# -------------------------------
def create_sequences(data, seq_length=192):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 3])  # predict "close" price (index 3)
    return np.array(X), np.array(y)

SEQ_LENGTH = 96 * 6  # 6 days

X_all, y_all = create_sequences(feature_data_scaled, SEQ_LENGTH)

train_size = int(len(X_all) * 0.8)
X_train, y_train = X_all[:train_size], y_all[:train_size]
X_test, y_test = X_all[train_size:], y_all[train_size:]

# -------------------------------
# 🧠 DEFINE LIGHT TRANSFORMER MODEL
# -------------------------------
def transformer_encoder(inputs, num_heads=4, d_model=64, dropout_rate=0.1):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization()(attention_output)
    return attention_output

def build_light_transformer(input_shape, num_layers=2):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_layers):
        x = transformer_encoder(x)
    outputs = x[:, -1, 0]  # Use last timestep's first feature ("open") as output
    model = Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="mse",
        metrics=["mae"]
    )
    return model

input_shape = (SEQ_LENGTH, X_train.shape[2])
model = build_light_transformer(input_shape)
model.summary()

# -------------------------------
# 🚀 TRAINING
# -------------------------------
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# -------------------------------
# 💾 SAVE MODEL IN .keras FORMAT
# -------------------------------
save_path = "model/btc_only/transformer_model_6days.keras"
model.save(save_path)
print(f"✅ Model saved to {save_path}")
