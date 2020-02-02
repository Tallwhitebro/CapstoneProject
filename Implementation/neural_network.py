import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
data_dir = '../cleaned_data/allStudents/'
data_files = os.listdir(data_dir)
dataset = np.loadtxt(data_dir + 'allStudents_10.csv', delimiter=',', skiprows=1)
print(dataset.shape)
for file in data_files:
	if file == 'allStudents_17.csv' or file == 'allStudents_10.csv':
		continue
	else:
		tmp = np.loadtxt(data_dir + file, delimiter=',', skiprows=1)
		print(file + ': ' + str(tmp.shape))
		dataset = np.append(dataset, tmp, axis=0)
		print(dataset.shape)

print('final')
print(dataset.shape)

# split into input (X) and output (y) variables
X = dataset[:,0:7]
y = dataset[:,7]

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=7, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)

#load test data
test_dataset = np.loadtxt(data_dir + 'allStudents_17.csv', delimiter=',', skiprows=1)
test_X = test_dataset[:,0:7]
test_y = test_dataset[:,7]

# evaluate the keras model
_, accuracy_model = model.evaluate(X, y)
print('Accuracy of model on dataset: %.2f' % (accuracy_model * 100))

# evaluate the keras model
_, accuracy_algorithm = model.evaluate(test_X, test_y)
print('Accuracy of algorithm on test data: %.2f' % (accuracy_algorithm * 100))
