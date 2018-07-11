import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

norm = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = norm.fit_transform(training_set)

x_train, y_train = [], []
for i  in range(60, 1258):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.expand_dims(x_train, axis=2)

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(rate=0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(LSTM(units=50))
model.add(Dropout(rate=0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=32, epochs=100)

# Predictions
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values.reshape(-1, 1)
inputs = norm.transform(inputs)

x_test = []
for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.expand_dims(x_test, axis=2)

predicted_stock_price = model.predict(x_test)
predicted_stock_price = norm.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color='green', label = 'Real')
plt.plot(predicted_stock_price, color='red', label = 'Pred')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('')
plt.legend()
plt.show()

