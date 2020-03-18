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

with open('data/DistanceCalculated.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

# print(data)

d = {}
for val in data:
    # print(val[3])
    try:
        d[val[0]] = float(val[3])
    except ValueError:
        pass

# print(d)

# using apply function to create a new column 
# df['Discounted_Price'] = df.apply(lambda row: row.Cost - 
#                                   (row.Cost * 0.1), axis = 1) 

years = [10,11,12,13,14,15,16,17]
for y in years:
    # YEAR:
    year = y
    fname = "allStudents_"+str(year)+".pkl"
    allStudents = load_pkl("pickles/"+fname)

    # print(allStudents["ZIP3"].head(50))
    allStudents['DIST'] = allStudents.apply(lambda row: d[row.ZIP3] if row.ZIP3 in d else np.nan, axis = 1)

    allStudents.to_csv('cleaned_data/allStudents/allStudents_'+str(year)+'.csv',index=False)

# print(allStudents.head(20))

# print("Distances:")
# print(allStudents[(allStudents["ACCEPTED"] == 1)]['DIST'].mean())
# print(allStudents[(allStudents["ACCEPTED"] == 0)]['DIST'].mean())
# print()

# print("Preferences:")
# print(allStudents[(allStudents["ACCEPTED"] == 1)]['PREF'].mean())
# print(allStudents[(allStudents["ACCEPTED"] == 0)]['PREF'].mean())


# print("\nMax values for each column:")

