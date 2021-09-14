'''
Import the required python and ML libraries.
'''
import os
import pandas as pd
import numpy as np
import datetime as dt
import scipy.io
import logging

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from math import sqrt

f = open("nopol.txt", "r")
Battery_name = f.readline()[0:]
index = int(f.readline()[0:])
pred_ahead = int(f.readline()[0:])
f.close()

mat_data = Battery_name[0:5] + '.mat'
mat = scipy.io.loadmat('CoE_battery_data\\' + mat_data)
mat = {k:v for k, v in mat.items() if k[0] != '_'}
raw_data = mat[Battery_name[0:5]]

class NasaCoEData():
    def __init__(self, data_file):
        self.raw_data = data_file
        self.logger = logging.getLogger()  
    def _data_extract(self):
        self.logger.info("Extracting revelant information")
        counter = 0
        dataset = []
        capacity_data = []
        for i in range(len(self.raw_data[0, 0]['cycle'][0])):
            row = self.raw_data[0, 0]['cycle'][0, i]
            if row['type'][0] == 'discharge':
                ambient_temperature = row['ambient_temperature'][0][0]
                date_time = dt.datetime(int(row['time'][0][0]),
                               int(row['time'][0][1]),
                               int(row['time'][0][2]),
                               int(row['time'][0][3]),
                               int(row['time'][0][4])) + dt.timedelta(seconds=int(row['time'][0][5]))
                data = row['data']
                capacity = data[0][0]['Capacity'][0][0]
                for j in range(len(data[0][0]['Voltage_measured'][0])):
                    voltage_measured = data[0][0]['Voltage_measured'][0][j]
                    current_measured = data[0][0]['Current_measured'][0][j]
                    temperature_measured = data[0][0]['Temperature_measured'][0][j]
                    current_load = data[0][0]['Current_load'][0][j]
                    voltage_load = data[0][0]['Voltage_load'][0][j]
                    time = data[0][0]['Time'][0][j]
                    dataset.append([counter + 1, ambient_temperature, date_time, capacity,
                    voltage_measured, current_measured,
                        temperature_measured, current_load,
                        voltage_load, time])
                capacity_data.append([counter + 1, ambient_temperature, date_time, voltage_measured, current_measured, 
                                      capacity])
                counter = counter + 1
        return [pd.DataFrame(data=dataset,
                       columns=['cycle', 'ambient_temperature', 'datetime',
                                'capacity', 'voltage_measured',
                                'current_measured', 'temperature_measured',
                                'current_load', 'voltage_load', 'time']),
          pd.DataFrame(data=capacity_data,
                       columns=['cycle', 'ambient_temperature', 'datetime', 'voltage_measured', 'current_measured',
                                'capacity'])]
NASA_data = NasaCoEData(raw_data)
dataset, capacity_data = NASA_data._data_extract()

n_steps = 5
n_pred = 3
def data_restructure(feature_data, label, n_steps, n_pred, index):
    X, y = list(), list()
    for i in range(index):
        end_ix = i+ n_steps
        
        if end_ix > index - n_pred:
            break
        
        seq_x, seq_y = feature_data[i:end_ix], label[end_ix: end_ix + n_pred]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def build_model(data, n_steps, n_pred, index, verbose, epochs, 
               batch_size):
    feature_data = data['capacity'][0:]#data['voltage_measured'][0:]
    label = data['capacity'][0:]
    X, Y = data_restructure(feature_data, label, n_steps, n_pred, index)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    
    n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], Y.shape[1]
    # model
    model = Sequential()
    #model.add(LSTM((n_outputs), batch_input_shape = (None, X.shape[1], X.shape[2]), return_sequences = False))
    model.add(LSTM(50, activation = 'relu', input_shape = (n_timesteps, n_features)))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(n_outputs))
    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy']) # 'mean_absolute_error'
    model.summary()
    # fit model
    model.fit(X,Y, epochs = epochs, batch_size = batch_size,
              verbose = verbose)
    return model

# Make a forecast
def forecast(model, new_observation, n_steps, verbose):
    new_observation = np.array(new_observation).reshape((1, new_observation.shape[0], 1))
    yhat = model.predict(new_observation, verbose = verbose)
    return yhat
    
# Model evaluation
def evaluate_model(data, n_steps, n_pred, index, pred_ahead, verbose, epochs, 
               batch_size):
    # fit model
    model = build_model(data, n_steps, n_pred, index, verbose, epochs, 
               batch_size)
    
    X_test, Y_test = list(), list()
    for j in range(pred_ahead):
        x_test_seq = data['capacity'][0:][index - n_steps - n_pred + j: index-n_pred + j]
        #data['voltage_measured'][0:][index - n_steps - n_pred +1 + j: index-n_pred + 1 + j]
        y_test_seq = data['capacity'][0:][index - n_pred + 1 + j: index + 1 + j]
        X_test.append(x_test_seq)
        Y_test.append(y_test_seq)
    
    predictions = list()
    for i in range(len(X_test)):
        yhat_sequence = forecast(model, X_test[i], n_steps, verbose)
        predictions.append(yhat_sequence)
    p = np.array([predictions[i][0] for i in range(len(predictions))])
    predictions = np.array(p)
    return predictions, np.array(X_test), np.array(Y_test)

def evaluate_forecasts(actual, predicted):
    scores = {
        k: 0 for k in range(actual.shape[1])
    }
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # Calculate MSE
        for j in range(actual.shape[0]):
            scores[i] += (actual[j,i]- predicted[j,i])**2
        # Calculate RMSE
        scores[i] = sqrt(scores[i])/actual.shape[0]
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s+= (actual[row, col] - predicted[row, col])**2
    score = sqrt(s/ (actual.shape[0]* actual.shape[1]))
    return score, scores


verbose, epochs, batch_size = 0, 1000, 32
predictions, X_test, Y_test = evaluate_model(capacity_data, n_steps, n_pred, index, pred_ahead, verbose, epochs, 
               batch_size)

score, scores = evaluate_forecasts(Y_test, predictions)

predictions_ = np.array([predictions[i][-1] for i in range(len(predictions))]) 
Y_test_ = np.array([Y_test[i][-1] for i in range(len(Y_test))])

capacity_actual = np.append(capacity_data['capacity'][0:][: index - n_pred], Y_test_)
capacity_pred = np.append(capacity_data['capacity'][0:][: index - n_pred], predictions_)

capacity_actual_zoom = np.append(capacity_data['capacity'][0:][index-1: index - n_pred], Y_test_)
capacity_pred_zoom = np.append(capacity_data['capacity'][0:][index-1: index - n_pred], predictions_)
