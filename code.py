""" Preprocessing the data """

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Loading the data
dataset_train = pd.read_csv("train.csv")
train = dataset_train.iloc[:, 1:2].values

# Feature Scaling
scaler = MinMaxScaler(feature_range = (0, 1))
train_scaled = scaler.fit_transform(train)

# Preparing the data
X_train = []
y_train = []
for i in range(30, len(train_scaled)):
    X_train.append(train_scaled[i-30 : i, 0])
    y_train.append(train_scaled[i, 0])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


""" Building the model """

# Importing the libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

model = Sequential()
model.add(LSTM(units = 30, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.1))

model.add(LSTM(units = 30, return_sequences = True))
model.add(Dropout(0.1))

model.add(LSTM(units = 30, return_sequences = True))
model.add(Dropout(0.1))

model.add(LSTM(units = 30))
model.add(Dropout(0.1))

model.add(Dense(units = 1))

model.compile(optimizer = "Adam", loss = "mean_squared_error")
model.fit(X_train, y_train, epochs = 100, batch_size = 30)


""" Making the predictions and visualising the results """

# Getting the real stock price
dataset_test = pd.read_csv("test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price
dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 30:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
X_test = []
for i in range(30, 50):
    X_test.append(inputs[i-30:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

model.save("model.h5")
