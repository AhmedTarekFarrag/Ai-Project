import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Read the dataset
df = pd.read_csv('/content/creditcard.csv')

# Drop the 'Time' column and remove missing values
df = df.drop(['Time'], axis=1)
df.dropna(inplace=True)

# Separate features and target variable
y = df['Class']
X = df.drop(['Class'], axis=1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# Scaling the data
scaler = RobustScaler()  # Using RobustScaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Autoencoder

# Encoder
input_layer = Input(shape=(X_train_scaled.shape[1],))
encoder = Dense(20, activation="relu")(input_layer)
encoder = Dense(15, activation="relu")(encoder)

# Decoder
decoder = Dense(20, activation="relu")(encoder)
decoder = Dense(X_train_scaled.shape[1], activation='sigmoid')(decoder)  # Using sigmoid activation for output layer

autoencoder = Model(inputs=input_layer, outputs=decoder)

# Encoder model
Encoder = Model(inputs=input_layer, outputs=encoder)

optimizer = keras.optimizers.RMSprop(learning_rate=0.1)
autoencoder.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer=optimizer)

history = autoencoder.fit(X_train_scaled, X_train_scaled, epochs=25, batch_size=32, validation_data=(X_test_scaled, X_test_scaled)).history

# Predicting train and test data on encoder
X_train_encoded = Encoder.predict(X_train_scaled)
X_test_encoded = Encoder.predict(X_test_scaled)

# KMeans for classification
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train_encoded)

# Predicting clusters for test data
yhat_train = kmeans.predict(X_train_encoded)
yhat_test = kmeans.predict(X_test_encoded)

# Map cluster labels to actual labels
yhat_train_mapped = np.array([1 if label == 1 else 0 for label in yhat_train])
yhat_test_mapped = np.array([1 if label == 1 else 0 for label in yhat_test])

# Calculate accuracy based on clustering
acc_train = accuracy_score(y_train, yhat_train_mapped)
acc_test = accuracy_score(y_test, yhat_test_mapped)

print("Train Accuracy:", acc_train)
print("Test Accuracy:", acc_test)
