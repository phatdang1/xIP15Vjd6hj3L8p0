from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
import keras.models
import pandas as pd
import numpy as np

def build_and_train_LSTM(x_train, Y_train):
    # LSTM Model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)]
    )

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, Y_train, epochs=25, batch_size=32)
    model.save('model')
    return model
    
def load_pretrained_model():
    model = keras.models.load_model('model')
    print(model.summary())
    return model

def test_model_plot(actual_prices, predicted_prices):

    #plot the test predictions
    plt.plot(actual_prices, color="black", label="Actual Price")
    plt.plot(predicted_prices, color="green", label="Predicted Price")
    plt.title('Share Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()