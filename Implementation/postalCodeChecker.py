# Imports
import os
import csv
import numpy as np
import pandas as pd
from pandas import Int64Index
import pickle
import sys
import math

def save_as_pkl(object, path):
    pickle.dump(object, open(path, "wb"))

def load_pkl(path):
    obj = pickle.load(open(path, "rb"))
    return obj

# Processing the .csvs into dataframes and saving them as pickles for easier
# loading on future runs
files = [f for f in os.listdir("cleaned_data/allStudents/")]
for file in files:
    # print ('\n'+file)
    fName = file.split('.')[0]
    df = pd.read_csv('cleaned_data/allStudents/' + file, delimiter=',', na_values=['NA'])
#     df["Id"] = df.reset_index().index
#     df.set_index("Id")
    # print(df.shape)
    save_as_pkl(df, 'pickles/'+fName+'.pkl')


postalCodes = []

years = [10,11,12,13,14,15,16,17]
for y in years:
    # YEAR:
    year = y
    fname = "allStudents_"+str(year)+".pkl"
    allStudents = load_pkl("pickles/"+fname)

    noNas = allStudents[allStudents["ZIP3"].notna()]["ZIP3"]

    l = set(list(noNas.to_numpy()))
    print(len(l), type(l), len(set(l)))

    for val in l:
        if val not in postalCodes:
            postalCodes.append(val)

# print(postalCodes, len(postalCodes))
    # In[ ]:

with open("postalCodes.csv", 'w', newline='') as myfile:
     for val in postalCodes:
        myfile.write(val + '\n')
