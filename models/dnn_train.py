import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tf_keras as keras 
from tf_keras.optimizers import Adam
from tf_keras.callbacks import EarlyStopping
import joblib

# ==========================================
# 1. LOAD & PREPARE DATA (4 FEATURES)
# ==========================================
# Make sure the file name matches your Java output
df = pd.read_csv('../real_dnn_training.csv') 

# Dropped 'dataSize' - keeping only the most impactful metrics
features = ['requiredCycles', 'serverId', 'edgeUtilization', 'wanBW']
X = df[features].values
y = df['actualDelay'].values

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data (Fit ONLY on training data!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. BUILD ADVANCED DNN ARCHITECTURE
# ==========================================
# Kept the 64 -> 32 -> 1 structure so your Java code doesn't need new layers,
# but changed input_dim to 4!
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_dim=4), # Expanded to 128
    keras.layers.Dense(64, activation='relu'),               # Added a middle layer
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

# Custom Optimizer: Slower learning rate for more precise weight adjustments
custom_adam = Adam(learning_rate=0.0005)
model.compile(optimizer=custom_adam, loss='mse', metrics=['mae'])

# ==========================================
# 3. TRAIN WITH EARLY STOPPING
# ==========================================
print("Training Optimized 4-Feature DNN...")

# Early stopping prevents overfitting. It stops training if the model 
# doesn't improve for 10 epochs, and restores the best weights.
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# We can safely set epochs high (150) because EarlyStopping will cut it off when it peaks
history = model.fit(
    X_train_scaled, y_train, 
    epochs=150, 
    batch_size=64, 
    validation_split=0.1, # Uses 10% of training data to check for early stopping
    callbacks=[early_stop], 
    verbose=1
)

# ==========================================
# 4. EVALUATE ACCURACY ON 20% TEST SET
# ==========================================
y_pred = model.predict(X_test_scaled).flatten()
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*45)
print("🎯 FINAL MODEL ACCURACY REPORT (TEST SET)")
print("="*45)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f} seconds")
print(f"R-squared (Overall Accuracy): {r2:.4f} (1.0 is perfect)")
print("="*45 + "\n")

# ==========================================
# 5. EXPORT WEIGHTS FOR JAVA S-HEO
# ==========================================
if not os.path.exists('param'): 
    os.makedirs('param')

weights = model.get_weights()
for i, w in enumerate(weights):
    np.savetxt(f"param/param_{i}.csv", w.flatten(), delimiter=",")

joblib.dump(scaler, 'param/scaler.pkl')
print("✅ 6 Weight files (param_0 to param_5) saved for Java!")