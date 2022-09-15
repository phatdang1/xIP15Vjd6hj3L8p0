import pandas_datareader as pddr
import datetime as dt
import numpy as np
import pandas as pd


# Load data from yahoo finance
def load_data_online(company:str, source:str, start:dt.datetime, end:dt.datetime):
    return pddr.DataReader(company, source, start, end)

def load_train_data(scale_data, prediction_days:int):
    x_train = []
    Y_train = []

    for x in range(70, len(scale_data)):
        x_train.append(scale_data[x-prediction_days: x, 0])
        Y_train.append(scale_data[x,0])

    x_train, Y_train = np.array(x_train), np.array(Y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, Y_train

    
def load_csv(filename):
    df = pd.read_csv(filename)
    #prepare data
    df = df[['Date','Price', 'Open', 'High', 'Low']]
    df['Open'] = df['Open'].str.replace(',','')
    df['Price'] = df['Price'].str.replace(',','')
    df['High'] = df['High'].str.replace(',','')
    df['Low'] = df['Low'].str.replace(',','')
    df.drop(df.tail(1).index,inplace=True)

    df['Open'] = df['Open'].astype('int')
    df['Price'] = df['Price'].astype('int')
    df['High'] = df['High'].astype('int')
    df['Low'] = df['Low'].astype('int')
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')

    open_df = df['Open']
    test_df = np.array(open_df[:int(open_df.shape[0]*0.2)])
    train_df = np.array(open_df[int(open_df.shape[0]*0.2):])
    print("test",train_df)
    #scale_data = scaler.fit_transform(train_df.reshape(-1,1))
