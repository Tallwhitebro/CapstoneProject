#!/usr/bin/env python
# coding: utf-8

# # Initial Data Analysis - Capstone Project

# In[1]:


# Imports
import os
import csv
import numpy as np
import pandas as pd
from pandas import Int64Index
import pickle
import sys
import math


# In[2]:


def save_as_pkl(object, path):
	pickle.dump(object, open(path, "wb"))

def load_pkl(path):
	obj = pickle.load(open(path, "rb"))
	return obj


# ### Preprocessing and loading the data

# In[3]:


# Processing the .csvs into dataframes and saving them as pickles for easier
# loading on future runs
files = [f for f in os.listdir("data/") if f.split('1')[0] == 'file']
for file in files:
    print ('\n'+file)
    fName = file.split('.')[0]
    df = pd.read_csv('data/' + file, delimiter=',', na_values=['NA'])
#     df["Id"] = df.reset_index().index
#     df.set_index("Id")
    print(df.shape)
    save_as_pkl(df, 'pickles/'+fName+'.pkl')


# In[4]:


## Loading preprocessed dataframes
pklFiles = [f for f in os.listdir("pickles/") if f.split('1')[0] == 'file']
for file in pklFiles:
	df = load_pkl("pickles/" + file)
	print(df.shape)


# # Change Year Loaded Here:

# In[5]:


# YEAR:
fname = "file15.pkl"
df = load_pkl("pickles/"+fname)


# ### Starting to analyze the data

# In[6]:


# First 30 columns:
firstPart = df.iloc[:,0:30]
# print(firstPart.columns.values)

# Columns 30 to 65: (top 6 course marks)
top6CourseMarks = df.iloc[:,30:66]
# print(top6CourseMarks.columns.values)

# cols = []
# for column in top6CourseMarks[top6CourseMarks.columns[2::3]]:
#     cols.append(df[column])
# hstack = pd.concat([x for x in cols], axis=1)
# hstack.fillna(0, inplace=True)
# hstack['Sum'] = hstack.mean(axis=1)
# print(hstack)

# Columns 66 to 71:
middle = df.iloc[:,66:72]
print(middle["WAVERG1"], middle["WAVERG2"])
# print(middle.columns.values)

# Columns 72 to 391:
choices = df.iloc[:, 72:]
# print(choices.columns.values)


# In[7]:


print(top6CourseMarks.head())
print(top6CourseMarks.columns)


# In[8]:


courseCodeCols = ['SECORCOD' + str(i+1) for i in range(12)]
maxScore = ['SECORC' + str(i+1) for i in range(12)] #I'm assuming that's what this column is? (useless)
studentScoreCols = ['SECORM' + str(i+1) for i in range(12)]
goalCourses = ['MHF4U', 'MCV4U', 'ENG4U', 'SCH4U', 'SPH4U']

# Advanced functions: MHF4U
# Calculus: MCV4U
# English: ENG4U
# Chemistry: SCH4U
# Physics: SPH4U
# Next highest mark, anything


# In[58]:


def averageFinder(allStudentMarks):
    studentAverages = []
    for index, row in allStudentMarks.iterrows(): 
        # Fancy syntax for extracting student's courses and grades they submitted (works for any number):
        studentCourseCodes = [row[courseCodeCol] for courseCodeCol in courseCodeCols]
        studentGrades = [row[studentScoreCol] for studentScoreCol in studentScoreCols]
        studentCourseCodes = [val for val in studentCourseCodes if type(val) == str]
        studentGrades = [val for val in studentGrades if str(val) != 'nan']
        
        necessaryCourses = [studentGrades[i] for i in range(len(studentGrades)) if studentCourseCodes[i] in goalCourses]
        remainder = [studentGrades[i] for i in range(len(studentGrades)) if studentCourseCodes[i] not in goalCourses]
        
        # Some students don't seem to have 6 courses.
        # Set their average to -1 in this case.
        try:
            average = (max(remainder) + sum(necessaryCourses))/6
            studentAverages.append(average)        
        except ValueError:
#             print("Student with incorrect number of courses applied.")
            studentAverages.append(-1)
    return studentAverages
    
# print(averageFinder(top6CourseMarks))


# In[59]:


# Finding the student's choice preference for the university:
# This data is found in the 'choices' dataframe.
uniChoiceCols = ['DUNI'+str(i+1) for i in range(20)]
programChoiceCols = ['DPRO1' + str(i+1) for i in range(20)]
goalUni = '196'


# In[60]:


def uniChoiceFinder(studentUniChoices):
    studentAverages = []
    for index, row in studentUniChoices.iterrows():
        # Filtering student's preference for our university and our specific program.
        studentCourseCodes = [i+1 for i in range(20) if (row['DUNI'+str(i+1)] == 196
                                                                                and (row['DPRO'+str(i+1)] == 'SIA' 
                                                                                or row['DPRO'+str(i+1)] == 'SI'))]
        studentAverages.append(studentCourseCodes[0])
    return studentAverages


# In[61]:


def acceptedOurUni(firstPart):
    acceptedArray = []
    for index, row in firstPart.iterrows():
        # Filtering student's preference for our university and our specific program.
        studentAccepted = int(row['CONFUNI'] == 196)
        acceptedArray.append(studentAccepted)
    return acceptedArray


# # Determining grade cutoff

# ### Seperating relevant data
# - Splitting data into students who accept and offer from mac and those who don't.
# - Isolating 8 initial columns of interest for analysis

# In[72]:


## Starting columns of interest:
# RESPROV - Province of Residence
# RESCNTY - County of Residence
# ZIP3 - Residence Postal Code (First 3 Digits)
# CONFUNI - Confirmed University (OurUni='196')
# CONFPR - Confirmed Program (OurProg='SI', OurProg_coop='SIA')
# CONFCHOIC - OUAC Confirmed Choice Preference
# WAVERG1 - Weighted Average (best 6 OAC / Senior Level current year finals)
# WAVERG2 - Weighted Average (best 6 OAC / Senior Level all year finals)
## First 6 columns are in first half, last 2 are in "Middle" dataframe.

COIsFirstHalf = ["GEND", "SCHOOL", "ZIP3", "CONFUNI", "CONFPR", "CONFCHOIC"]
COIsSecondHalf = ["WAVERG2"]

# Finding the student's average from the formula given by Dr. Franek.
averages = averageFinder(top6CourseMarks)
averagesDf = pd.DataFrame(averages, columns=['AVG'])

# Making a dataframe with student's preferences
preferences = uniChoiceFinder(choices)
preferencesDf = pd.DataFrame(preferences, columns=['PREF'])

# Dataframe with 1 for accepted our uni, 0 otherwise.
acceptedArray = acceptedOurUni(firstPart)
acceptedDf = pd.DataFrame(acceptedArray, columns=['ACCEPTED'])

# Adding columns of interest found in first half of the data
columnsOfInterest = pd.concat([firstPart[x] for x in COIsFirstHalf], axis=1)

# Adding the rest of the columns of interest
columnsOfInterest = pd.concat([columnsOfInterest] + [middle[x] for x in COIsSecondHalf] + 
                              [averagesDf] + [preferencesDf] + [acceptedDf],axis=1)
copy = columnsOfInterest.copy()

mcmasterVector = copy["CONFUNI"] == 196
notmcmasterVector = copy["CONFUNI"] != 196
allStudents = copy


# ### Some min/max/average output for both student types:

# In[68]:


# Students that received and accepted a McMaster offer:
acceptedMcMaster = copy[mcmasterVector]
# print(acceptedMcMaster)
# print(acceptedMcMaster[(acceptedMcMaster["WAVERG2"] > 0) & (acceptedMcMaster["CONFPR"] == "SIA")].min())
print("Students who accepted an offer from McMaster:")
print("Mean values for each column:")
print(acceptedMcMaster[(acceptedMcMaster["AVG"] > 0) & (acceptedMcMaster["CONFPR"] == "SIA")].mean())
print("\nMax values for each column:")
print(acceptedMcMaster[(acceptedMcMaster["AVG"] > 0) & (acceptedMcMaster["CONFPR"] == "SIA")].max())
print("\nLowest 5 weighted average:")
print(acceptedMcMaster[(acceptedMcMaster["AVG"] > 0) & (acceptedMcMaster["CONFPR"] == "SI")].nsmallest(20,"WAVERG2"))
print("\nShape of acceptedMcMaster df (number of students that accepted):")
print(acceptedMcMaster.shape)

# Students that didn't accept a mcmaster offer:
print('\n\n\n\n')
notMcMaster = copy[notmcmasterVector]
# print(notMcMaster)
print("Min average of students who didn't accept mac offer")
print(notMcMaster[notMcMaster["AVG"] > 0].min())
print("\nMax average of students who didn't accept mac offer")
print(notMcMaster[notMcMaster["AVG"] > 0].max())
print("\nAverage average of students who didn't accept mac offer")
print(notMcMaster[notMcMaster["AVG"] > 0].mean())


# In[77]:


# print(acceptedMcMaster.shape)
# print(notMcMaster.shape)
print('McMaster students sample:')
print(acceptedMcMaster.head(10))
print('\n\n')
print('All Students sample:')
print(allStudents.head(20))
# print(notMcMaster.head(50))

