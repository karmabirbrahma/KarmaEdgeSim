import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tf_keras as keras 
import joblib

# Load data
df = pd.read_csv('../dnn_training_data.csv')
features = ['alpha', 'beta', 'requiredCycles', 'dataSize', 'serverId', 'queueLength', 'channelRate']
X = df[features].values
y = df[['actualDelay', 'actualEnergy']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build & Train
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_dim=7),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2, activation='linear')
])
model.compile(optimizer='adam', loss='mse')
print("Training DNN...")
model.fit(X_scaled, y, epochs=50, batch_size=64, verbose=1)

# SAVE AS CSV (Cleaner for Java to parse)
if not os.path.exists('models'): os.makedirs('models')

weights = model.get_weights()
for i, w in enumerate(weights):
    # We save as comma-separated values
    np.savetxt(f"models/param_{i}.csv", w.flatten(), delimiter=",")

joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Weights saved as .csv files in models/")
