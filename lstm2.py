import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load your dataset
file_path = '/Users/taylor/Desktop/powerball.csv'
data = pd.read_csv(file_path)

# Assuming you're predicting all the main numbers and Powerball
# You need to ensure your dataset is structured appropriately for this
data = data[['num_1', 'num_2', 'num_3', 'num_4', 'num_5', 'powerball']].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Split the data into training and test sets
train_size = int(len(data) * 0.67)
test_size = len(data) - train_size
train, test = data[0:train_size, :], data[train_size:len(data), :]

# Convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, :])
    return np.array(X), np.array(Y)

# Reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 6))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 6))

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 6)))  # Adjusted for 6 features
model.add(Dense(6))  # Output layer to predict 6 numbers
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# Making predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Inverse transform predictions and target values to their original scale
trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)

# # Convert predictions from continuous float to discrete numbers and print them
# print("Predicted training numbers:")
# for i, prediction in enumerate(trainPredict):
#     print(f"Draw {i+1}: num_1={int(round(prediction[0]))}, num_2={int(round(prediction[1]))}, num_3={int(round(prediction[2]))}, num_4={int(round(prediction[3]))}, num_5={int(round(prediction[4]))}, powerball={int(round(prediction[5]))}")

# print("\nPredicted testing numbers:")
# for i, prediction in enumerate(testPredict):
#     print(f"Draw {i+1}: num_1={int(round(prediction[0]))}, num_2={int(round(prediction[1]))}, num_3={int(round(prediction[2]))}, num_4={int(round(prediction[3]))}, num_5={int(round(prediction[4]))}, powerball={int(round(prediction[5]))}")

last_prediction = testPredict[-1]

# Round the predictions to the nearest integer and print the most likely next draw
print("Most likely next draw:")
print(f"num_1={int(round(last_prediction[0]))}, num_2={int(round(last_prediction[1]))}, num_3={int(round(last_prediction[2]))}, num_4={int(round(last_prediction[3]))}, num_5={int(round(last_prediction[4]))}, powerball={int(round(last_prediction[5]))}")