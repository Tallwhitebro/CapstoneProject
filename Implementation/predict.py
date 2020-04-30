import neural_network
import support_vector_machine
import data
import util
import numpy as np
import pandas as pd

def predict(model, file_to_path):
    _, X_test, _, _ = data.retrieve_data()

    # predicting results
    print("PREDICTION: ")
    y_predict = model.predict(X_test)
    unique, counts = np.unique(y_predict, return_counts=True)
    print(dict(zip(unique, counts)))
    
    util.save_as_pkl(y_predict, file_to_path)

def predict_NN():
    model = neural_network.retrieve_model()
    file_to_path = "../prediction/NN_prediction.pkl"

    predict(model, file_to_path)

def predict_SVM():
    model = support_vector_machine.retrieve_model()
    file_to_path = "../prediction/SVM_prediction.pkl"

    predict(model, file_to_path)

def allocate_students(sorted_students, max_capacity, model): #array of arrays, int
	accepted_students = 0 #int
	num_students = 0
	cutoff_avg = -1
	
	for index, student in sorted_students.iterrows():
		np_arr = student.to_numpy()
		np_arr = np_arr.reshape(1,-1) # Reformatting the np array to the proper shape
		student_avg = student['AVG']

		if model.predict(np_arr): # they will accept, extend offer
			accepted_students+=1

		if accepted_students == max_capacity:
			cutoff_avg = student_avg
			break
		num_students += 1
	
	return cutoff_avg

def calculate_cutoff(data_from_path, target_seat_cap, model):
	df = pd.read_csv(data_from_path)
	# Sorting the data by average
	df = df.sort_values(by=['AVG'], ascending = False)
	if "ACCEPTED" in df.keys():
		df = df.drop(['ACCEPTED'], axis='columns') # drop target column from dataset

	if "ZIP3" in df.keys():
		df = df.drop(['ZIP3'], axis='columns')

	cutoff_avg = allocate_students(df, target_seat_cap, model)

	print("To reach the seat goal of " + str(target_seat_cap) + " seats,")
	print("the cutoff average must be set to " + str(round(cutoff_avg, 1))+"%\n")

	return cutoff_avg

if __name__ == "__main__":
    model = neural_network.retrieve_model()
    calculate_cutoff("../cleaned_data/allStudents/allStudents_17.csv", 300, model)