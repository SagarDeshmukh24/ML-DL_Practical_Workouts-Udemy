# Add libraries to be used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


# Load the dataset
data = pd.read_csv('Car_Purchasing_Data.csv', encoding="ISO8859-1")


# Display the dataset

# print(data.head())
# sns.pairplot(data)
# plt.show()


# Data Cleaning and Preparation of training and testing sets
df=data.drop(['Customer Name', 'Customer e-mail', 'Country'], axis=1)
X=df.drop(['Car Purchase Amount'], axis=1)
y=df['Car Purchase Amount']


# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1,1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1,1))


# Model Building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')


# Model Training
history = model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=25, verbose=1, validation_split=0.2)


#Loss Curves
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='validation')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Training_loss', 'Validation_loss'], loc='upper right')
# plt.show()

# Create random data for prediction
random_data = np.array([[1, 50, 58000, 12000, 5000000]])
# random_data_scaled = scaler.transform(random_data)
predicted_price_scaled = model.predict(random_data)
# predicted_price = scaler.inverse_transform(predicted_price_scaled)
print(f"Predicted Car Purchase Amount: {predicted_price_scaled}")