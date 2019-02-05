import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

# Set the lowest value as 0 and highest value as 1
def MinMaxScaler(data):
     numerator = data - np.min(data, 0)
     denominator = np.max(data,0) - np.min(data,0)
     return numerator / (denominator + 1e-7), np.max(data,0), np.min(data,0)

# Return the scaled value to origin
def MinMaxReturn(data, max, min):
    data = pd.DataFrame(data)
    return data * (max - min + 1e-7) + min

# Create the dataset for training
def create_dataset(signal_data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(signal_data)-look_back):
        dataX.append(signal_data[i:(i+look_back), 0])
        dataY.append(signal_data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# CustomHistory for Training
class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))

look_back = 20
data_location = "international-airline-passengers.csv"

# Airline passengers data
passengers_data = pd.read_csv(data_location, header=None)

passengers_data, max, min = MinMaxScaler(passengers_data)

train = np.array(passengers_data[0:1500])
test = np.array(passengers_data[1500:])

x_train, y_train = create_dataset(train, look_back)
x_test, y_test = create_dataset(test, look_back)

# Change the dimension to 3d for training
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Create the model
model = Sequential()
model.add(LSTM(32, input_shape=(look_back, 1), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(32, input_shape=(look_back, 1), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(32, input_shape=(look_back, 1)))
model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
custom_hist = CustomHistory()
custom_hist.init()

for i in range(50):
    print("Epochs:", (i+1))
    model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False, callbacks=[custom_hist], verbose = 1)
    model.reset_states()

# Evaluate the model
trainScore = model.evaluate(x_train, y_train, batch_size=1, verbose=0)
model.reset_states()
print('Train Score: ', trainScore)
testScore = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
model.reset_states()
print('Test Score: ', testScore)

# Predict using the trained model
look_ahead = len(y_test)
xhat = x_test[0]
predictions = np.zeros((look_ahead, 1))

for i in range(look_ahead):
    prediction = model.predict(np.array([xhat]), batch_size=1)
    predictions[i] = prediction
    xhat = np.vstack([xhat[1:], prediction])

passengers_data = MinMaxReturn(passengers_data, max, min)
predictions = MinMaxReturn(predictions, max, min)

plt.figure(figsize=(12,5))
plt.axvline(x=len(train), color='r', linestyle='--')
plt.plot(np.arange(len(passengers_data)), passengers_data, 'b', label="test function")
plt.plot(np.arange(look_ahead) + len(train), predictions,'r', label="prediction")
plt.legend()
plt.show()
