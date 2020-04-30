import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split # for splitting dataset
import util
import data

def retrieve_model():
	model = util.load_pkl("../models/NN.pkl")
	return model

def create_model():
	X_train, _, y_train, _ = data.retrieve_data()

	# define the keras model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	# compile the keras model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	# fit the keras model on the dataset
	model.fit(X_train, y_train, epochs=150, batch_size=10)

	util.save_as_pkl(model, "../models/NN.pkl")

if __name__ == "__main__":
    create_model()