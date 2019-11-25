import os
import csv
import numpy as np
import pandas as pd
from pandas import Int64Index
import pickle
import sys

def save_as_pkl(object, path):
	pickle.dump(object, open(path, "wb"))

def load_pkl(path):
	obj = pickle.load(open(path, "rb"))
	return obj

# Processing the .csvs into dataframes and saving them as pickles for easier
# loading on future runs
files = [f for f in os.listdir("../data") if f.split('1')[0] == 'file']
for file in files:
	print ('\n'+file)
	fName = file.split('.')[0]
	df = pd.read_csv('../data/' + file, delimiter=',', na_values=['NA'])
	print(df)
	idx = Int64Index([range(0, len(df.index))])
	print(idx.shape)
	df.insert(0, "Id", idx)
	quit()
	print(df.shape)
	save_as_pkl(df, '../pickles/'+fName+'.pkl')
quit()

## Loading preprocessed dataframes
# pklFiles = [f for f in os.listdir("../pickles") if f.split('1')[0] == 'file']
# for file in pklFiles:
# 	df = load_pkl("../pickles/" + file)
# 	print(df.shape)

df = load_pkl("../pickles/file10.pkl")

# First 30 columns:
firstPart = df.iloc[:,0:30]
# print(firstPart.columns.values)

# Columns 30 to 65: (top 6 course marks)
top6CourseMarks = df.iloc[:,30:66]
# print(top6CourseMarks.columns.values)

cols = []
for column in top6CourseMarks[top6CourseMarks.columns[2::3]]:
    cols.append(column)

summed = []
for val in cols:
	summed += val

print(summed)

# Columns 66 to 71:
middle = df.iloc[:,66:72]
# print(middle.columns.values)

# COlumns 72 to 391:
choices = df.iloc[:, 72:]
# print(choices.columns.values)


