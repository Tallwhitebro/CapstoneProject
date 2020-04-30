# Imports
import os
import csv
import numpy as np
import pandas as pd
from pandas import Int64Index
import pickle
import sys
import math

# Save object into compressed pickle file
def save_as_pkl(object, path):
	pickle.dump(object, open(path, "wb"))

# Load from pickle file
def load_pkl(path):
	obj = pickle.load(open(path, "rb"))
	return obj


# Processing the .csvs into dataframes and saving them as pickles for easier
# loading on future runs
files = [f for f in os.listdir("data/") if f.split('1')[0] == 'file']
for file in files:
    print ('\n'+"Loading " + file)
    fName = file.split('.')[0]
    df = pd.read_csv('data/' + file, delimiter=',', na_values=['NA'])
#     df["Id"] = df.reset_index().index
#     df.set_index("Id")
    save_as_pkl(df, 'pickles/'+fName+'.pkl')

## Loading preprocessed dataframes
pklFiles = [f for f in os.listdir("pickles/") if f.split('1')[0] == 'file']
for file in pklFiles:
	df = load_pkl("pickles/" + file)

with open('data/DistanceCalculated.csv', newline='') as f:
    reader = csv.reader(f)
    distances = list(reader)

# Analyzing each year's data one at a time.
years = [10,11,12,13,14,15,16,17]
for y in years:
    year = y
    fname = "file"+str(year)+".pkl"
    df = load_pkl("pickles/"+fname)

    # First 30 columns:
    firstPart = df.iloc[:,0:30]
    # print(firstPart.columns.values)

    # Columns 30 to 65: (top 6 course marks)
    top6CourseMarks = df.iloc[:,30:66]

    # Columns 66 to 71:
    middle = df.iloc[:,66:72]

    # Columns 72 to 391:
    choices = df.iloc[:, 72:]

    courseCodeCols = ['SECORCOD' + str(i+1) for i in range(12)]
    maxScore = ['SECORC' + str(i+1) for i in range(12)]
    studentScoreCols = ['SECORM' + str(i+1) for i in range(12)]
    goalCourses = ['MHF4U', 'MCV4U', 'ENG4U', 'SCH4U', 'SPH4U']

    # Target courses:
    # Advanced functions: MHF4U
    # Calculus: MCV4U
    # English: ENG4U
    # Chemistry: SCH4U
    # Physics: SPH4U
    # Next highest mark, anything

    # Calculates student's averages given the 5 required courses for entry
    def averageFinder(allStudentMarks):
        studentAverages = []
        for index, row in allStudentMarks.iterrows(): 
            try:
                # The courses the student took
                studentCourseCodes = [row[courseCodeCol] for courseCodeCol in courseCodeCols]
                studentCourseCodes = [val for val in studentCourseCodes if type(val) == str]

                # The grades the student received in each course
                studentGrades = [row[studentScoreCol] for studentScoreCol in studentScoreCols]
                studentGrades = [val for val in studentGrades if str(val) != 'nan']

                # Required course
                necessaryCourses = [int(studentGrades[i]) for i in range(len(studentGrades)) if studentCourseCodes[i] in goalCourses]
                # Remaining course(s)
                remainder = [int(studentGrades[i]) for i in range(len(studentGrades)) if studentCourseCodes[i] not in goalCourses]

                # The number of courses the student applied with.
                # Some student's data seems to be inaccurate, they show
                # less than the 6 required courses.
                length = len(necessaryCourses) + min(len(remainder),1)
                average = (max(remainder) + sum(necessaryCourses))/length
                studentAverages.append(average)

            except ValueError:
                studentAverages.append(np.nan)

        return studentAverages

    # Finding the student's choice preference for the university:
    # This data is found in the 'choices' dataframe.
    uniChoiceCols = ['DUNI'+str(i+1) for i in range(20)]
    programChoiceCols = ['DPRO1' + str(i+1) for i in range(20)]
    goalUni = '196'

    # Determine which preference value the student 
    # gave to our target university
    def uniChoiceFinder(studentUniChoices):
        studentAverages = []
        for index, row in studentUniChoices.iterrows():
            # Filtering student's preference for our university and our specific program.
            studentCourseCodes = [i+1 for i in range(20) if (row['DUNI'+str(i+1)] == 196
                                                                    and (row['DPRO'+str(i+1)] == 'SIA' 
                                                                    or row['DPRO'+str(i+1)] == 'SI'))]
            studentAverages.append(studentCourseCodes[0])

        return studentAverages

    # Determine which student's received an offer
    # from the target university and accepted the offer.
    def acceptedOurUni(firstPart):
        acceptedArray = []
        for index, row in firstPart.iterrows():
            # Filtering student's preference for our university and our specific program.
            studentAccepted = int(row['CONFUNI'] == 196 and (row['CONFPR'] == 'SIA' or row['CONFPR'] == 'SI'))
            acceptedArray.append(studentAccepted)
        return acceptedArray

    # Columns of interest:
    # GEND - The student's gender
    # SCHOOL - The student's highschool ID
    # ZIP3 - Residence Postal Code (First 3 Digits)
    # CONFUNI - Confirmed University (OurUni='196')
    # CONFCHOIC - OUAC Confirmed Choice Preference
    # WAVERG2 - Weighted Average (best 6 OAC / Senior Level all year finals)
    # Note, WAVERG2 differs from the calculated average given the 5 required courses.

    ## First 6 columns are in first half, last 2 are in "Middle" dataframe.

    COIsFirstHalf = ["GEND", "SCHOOL", "ZIP3", "CONFUNI", "CONFCHOIC"]
    COIsSecondHalf = ["WAVERG2"]

    # Finding the student's average using the 5 required courses
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

    # Determining the list of zip codes
    zipCodes = []
    zipData = columnsOfInterest[columnsOfInterest["ZIP3"].notna()]["ZIP3"]
    zipSet = set(list(zipData.to_numpy()))
    for zipCode in zipSet:
        if zipCode not in zipCodes:
            zipCodes.append(zipCode)

    # For each zip code present, determine the 
    # corresponding distance from the target
    # university.
    distanceDict = {}
    for distance in distances:
        try:
            distanceDict[distance[0]] = float(distance[3])
        except ValueError:
            pass

    # For each row, add the corresponding distance from 
    # the unversity to a column labeled 'DIST'
    columnsOfInterest['DIST'] = columnsOfInterest.apply(lambda row: distanceDict[row.ZIP3]
            if row.ZIP3 in distanceDict else np.nan, axis = 1)

    del columnsOfInterest['ZIP3']
    copy = columnsOfInterest.copy()

    targetUni = copy["CONFUNI"] == 196
    notTargetUni = copy["CONFUNI"] != 196
    allStudents = copy

    # Students that received and accepted an offer
    # from the target university:
    acceptedTargetUni = copy[targetUni]

    # Calculating the student with the lowest average who accepted the target university.
    # This is used to calculate the cutoff point.
    smallest = acceptedTargetUni[(acceptedTargetUni["AVG"] > 0) & (acceptedTargetUni["ACCEPTED"] == 1)].nsmallest(1,"AVG").iloc[0]['AVG']
    # From this, generating the list of all students who received an offer.
    receivedOffer = acceptedTargetUni[(acceptedTargetUni["AVG"] >= smallest)]

    # # Students that didn't accept a received offer
    didntAccept = copy[notTargetUni]

    # Exporting the dataframes to CSV
    allStudents.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True) #dropping all rows that contain an empty value
    allStudents.to_csv('cleaned_data/allStudents/allStudents_'+str(year)+'.csv',index=False)
    acceptedTargetUni.to_csv('cleaned_data/acceptedOurUni/acceptedOurUni_'+str(year)+'.csv',index=False)
    didntAccept.to_csv('cleaned_data/didntAccept/didntAccept_'+str(year)+'.csv',index=False)
    receivedOffer.to_csv('cleaned_data/receivedOffer/receivedOffer_'+str(year)+'.csv',index=False)
    print("Finished processing year " + str(year))

