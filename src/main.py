import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
import pandas_datareader as pddr
from loading_data import load_data_online, load_train_data
from model import build_and_train_LSTM, test_model_plot, load_pretrained_model
from sklearn.preprocessing import MinMaxScaler

prediction_days = 70
company = 'AAPL'
source = 'yahoo'
# Load data
data = load_data_online(company, source, dt.date(2012,1,1), dt.date(2020,12,30))
# scale data into values between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
# tranform close data into scale values
scale_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
# Load train data
#x_train, Y_train = load_train_data(scale_data, prediction_days)
# Build or use pre-trained model
#model = build_and_train_LSTM(x_train, Y_train)
model = load_pretrained_model()
# Test model
test_start = dt.datetime(2020,12,30)
test_end = dt.datetime.now()
test_data = load_data_online(company, source, test_start, test_end)
actual_prices = test_data['Close'].values
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

# make prediction
x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

test_model_plot(actual_prices, predicted_prices)

# Predict Stock Price
data = [model_inputs[len(model_inputs) + 1 - prediction_days : len(model_inputs + 1), 0]]
data = np.array(data)
data = np.reshape(data, (data.shape[0], data.shape[1],1))

prediction = model.predict(data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction:{prediction}")