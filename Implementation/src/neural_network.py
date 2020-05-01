import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense
import util
import data

def retrieve_model():
	model = util.load_pkl("../models/NN.pkl")
	return model

def create_model():
	X_train, _, y_train, _ = data.retrieve_data()

	# define the keras model
	model = Sequential()
	model.add(Dense(100, input_dim=8, activation='relu'))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	# compile the keras model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	# fit the keras model on the dataset
	model.fit(X_train, y_train, epochs=50, batch_size=10)

	util.save_as_pkl(model, "../models/NN.pkl")

	return model

if __name__ == "__main__":
    create_model()