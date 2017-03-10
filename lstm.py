from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
from random import randint

data_dim = 132
timesteps = 1375
nb_classes = 2

model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


right_signals = pd.read_csv('W_into_right.csv',header=None)
left_signals = pd.read_csv('W_into_left.csv',header=None)
left_signals = reshape(left_signals,())
all_signals = []
lc=0
rc=0
for i in range(len(left_signals[0]) + len(right_signals[0])):
    rd = randint(0,50)
    if rd<= 25:
        if(lc== 66):
            all_signals.append(right_signals[:,(lc-1)*1375:lc*1375])
        all_signals.append(left_signals[(lc-1)*1375:lc*1375,:])
        lc++
    elif rd>25:
        if(rc==64):
            continue
        all_signals.append(right_signals(WRITE ACCESS HERE))
        rc++
