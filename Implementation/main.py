import pickle
import pandas as pd
import numpy as np

def MLAlgorithm(student = []):
	willAccept = False
	# run through ML algorithm, which will return whether or not student will accept
	return True

def allocateStudents(sortedStudents, maxCapacity, model): #array of arrays, int
	
	# print("Sorted list of students by averages:")
	# for x in range(len(sortedStudents)): 
	# 	print(sortedStudents[x])
	
	acceptedStudents = 0 #int
	nStudents = 0
	cutoffAvg = -1
	
	for index, student in sortedStudents.iterrows():
		npArr = student.to_numpy()
		npArr = npArr.reshape(1,-1) # Reformatting the np array to the proper shape
		studentAvg = student['AVG']

		if model.predict(npArr): # they will accept, extend offer
			acceptedStudents+=1

		if acceptedStudents == maxCapacity:
			cutoffAvg = studentAvg
			break
		nStudents += 1
	
	return cutoffAvg

# Load from pickle file
def load_pkl(path):
	obj = pickle.load(open(path, "rb"))
	return obj

def main():
	print("\nReading student data")
	# The location of the input data
	dataPath = "../cleaned_data/allStudents/allStudents_17.csv"
	df = pd.read_csv(dataPath)
	# Sorting the data by average
	df = df.sort_values(by=['AVG'], ascending = False)
	if "ACCEPTED" in df.keys():
		df = df.drop(['ACCEPTED'], axis='columns') # drop target column from dataset

	if "ZIP3" in df.keys():
		df = df.drop(['ZIP3'], axis='columns')

	print("Loading model")
	# Target number of university acceptances
	targetSeatCap = 300
	# Model choice location
	modelPath = "../models/SVM.pkl"
	# loading the model
	model = load_pkl(modelPath)

	print("Calculating cutoff\n")
	cutoffAvg = allocateStudents(df, targetSeatCap, model)

	print("To reach the seat goal of " + str(targetSeatCap) + " seats,")
	print("the cutoff average must be set to " + str(round(cutoffAvg,1))+"%\n")

if __name__ == "__main__":
	main()