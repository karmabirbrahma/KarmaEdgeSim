import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import joblib

df = pd.read_csv('../dnn_training_data.csv')
features = ['alpha', 'beta', 'requiredCycles', 'dataSize', 'serverId', 
            'queueLength', 'channelRate']

X = df[features].values
y = df[['actualDelay', 'actualEnergy']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(7,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='linear'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, verbose=1)

model.save('offline_dnn_model_keras2_final.h5', save_format='h5')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model saved successfully with Keras 2")
