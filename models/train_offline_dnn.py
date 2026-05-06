import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import joblib

print("Loading data...")
df = pd.read_csv('../.dnn_training_data.csv')

# Features we are using
features = ['alpha', 'beta', 'requiredCycles', 'dataSize', 'serverId', 
            'queueLength', 'channelRate']

X = df[features].values
y = df[['actualDelay', 'actualEnergy']].values   # two outputs to predict

print(f"Dataset shape: {X.shape} features → {y.shape} targets")

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build simple Multi-Layer Perceptron (MLP)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(2, activation='linear')        # predicts delay and energy
])

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='mse', 
              metrics=['mae'])

model.summary()

# Train the model
print("\nTraining the Offline DNN...")
history = model.fit(X_train, y_train, 
                    epochs=100, 
                    batch_size=64, 
                    validation_split=0.2, 
                    verbose=1)

# Evaluate
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Training finished! Test MAE: {mae:.4f}")

# Save model and scaler
model.save('offline_dnn_model.h5')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model saved as 'offline_dnn_model.h5'")
print("✅ Scaler saved as 'scaler.pkl'")
