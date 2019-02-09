import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

# Set the lowest value as 0 and highest value as 1
def MinMaxScaler(data, max, min):
     numerator = data - min
     denominator = max - min
     if denominator is 0:
         return 0, max, min
     return numerator / denominator

# Return the scaled value to origin
def MinMaxReturn(data, max, min):
    return data * (max - min) + min

# CustomHistory for Training
class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))

data_location = "data/google-stock.csv"

# Google Stock data
raw_data = pd.read_csv(data_location, header=0, usecols=[1,2,3,4])
raw_data = raw_data.iloc[::-1].values
max = raw_data.max()
min = raw_data.min()
google_stock_data = MinMaxScaler(raw_data, max, min)

train = np.array(google_stock_data[0:830])
test = np.array(google_stock_data[830:])
x_train = train[:, :3]
y_train = train[:, 3]
x_test = test[:, :3]
y_test = test[:, 3]

# Change dimensions
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))

# Create the model
model = Sequential()
model.add(LSTM(32, input_shape=(3,1), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(32, input_shape=(3,1), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(32, input_shape=(3,1)))
model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
custom_hist = CustomHistory()
custom_hist.init()

model.fit(x_train, y_train, epochs=100, batch_size=20, shuffle=False, callbacks=[custom_hist], verbose = 1)

# Evaluate the model
trainScore = model.evaluate(x_train, y_train, batch_size=20, verbose=0)
print('Train Score: ', trainScore)
testScore = model.evaluate(x_test, y_test, batch_size=20, verbose=0)
print('Test Score: ', testScore)

# Predict using the trained model
prediction = model.predict(x_test)
prediction = MinMaxReturn(prediction, max, min)

plt.figure(figsize=(12,5))
plt.axvline(x=len(train), color='r', linestyle='--')
plt.plot(np.arange(len(raw_data)), raw_data[:, 3], 'b', label="Raw Data")
plt.plot(np.arange(len(x_train),len(raw_data)), prediction,'r', label="Prediction")
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
