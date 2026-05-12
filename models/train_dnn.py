import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tf_keras as keras 
from tf_keras.optimizers import Adam
from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib

# ==========================================
# 1. LOAD & PREPARE DATA (4 FEATURES)
# ==========================================
df = pd.read_csv('../ulti_dnn_training_data.csv') 

features = ['requiredCycles', 'serverId', 'edgeUtilization', 'wanBW']
X = df[features].values
y = df['actualDelay'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. BUILD ADVANCED DNN ARCHITECTURE
# ==========================================
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_dim=4), # Expanded to 128
    keras.layers.Dense(64, activation='relu'),               # Added a middle layer
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

custom_adam = Adam(learning_rate=0.001) # Starting slightly higher so the Plateau reducer has room to work
model.compile(optimizer=custom_adam, loss='mse', metrics=['mae'])

# ==========================================
# 3. TRAIN WITH CALLBACKS (Early Stop + Reduce LR)
# ==========================================
print("Training Optimized 4-Feature DNN...")

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# THE SNIPER: If validation loss stalls for 5 epochs, cut learning rate by 50%
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

history = model.fit(
    X_train_scaled, y_train, 
    epochs=150, 
    batch_size=64, 
    validation_split=0.1, 
    callbacks=[early_stop, reduce_lr], 
    verbose=1
)

# ==========================================
# 4. PLOT TRAINING HISTORY
# ==========================================
print("\nGenerating Training History Graph...")
plt.figure(figsize=(12, 5))

# Plot 1: Mean Squared Error (Loss)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss (MSE) Over Time')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

# Plot 2: Mean Absolute Error
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE', color='blue')
plt.plot(history.history['val_mae'], label='Validation MAE', color='orange')
plt.title('Mean Absolute Error Over Time')
plt.ylabel('Error (Seconds)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("📈 Graph saved successfully as 'training_history.png'")

# ==========================================
# 5. EVALUATE ACCURACY ON 20% TEST SET
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
print(f"R-squared accu (Overall Accuracy): {r2:.4f} (1.0 is perfect)")
print("="*45 + "\n")

# ==========================================
# 6. EXPORT WEIGHTS FOR JAVA S-HEO
# ==========================================
if not os.path.exists('models'): 
    os.makedirs('models')

weights = model.get_weights()
for i, w in enumerate(weights):
    np.savetxt(f"models/param_{i}.csv", w.flatten(), delimiter=",")

joblib.dump(scaler, 'models/scaler.pkl')
print("✅ 6 Weight files (param_0 to param_7) saved for Java!")