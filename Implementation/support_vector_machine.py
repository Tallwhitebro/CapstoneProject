import numpy as np
import time # for timing training of data
import pandas as pd
from sklearn.svm import SVC # actual support vector classifier
from sklearn.model_selection import train_test_split # for splitting dataset

# load cleaned up csvs into dataframe
path = '../cleaned_data/receivedOffer/receivedOffer_'
years = [11,12,13,14,15,16, 17]
target = 'ACCEPTED'

df = pd.read_csv(path + '10.csv')

# add more data to dataset
for y in years:
    #print(' + Year: ' + str(y) + ' = ')
    df = df.append(pd.read_csv(path + str(y)+'.csv'))
    
print('Total students read in: ' + str(len(df)))

df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True) #dropping all rows that contain an empty value

print('Total students ready for ML alg: ' + str(len(df)))

#maybe combining and scrambling the data and splitting the data 80-20 for training and testing would be better?
#so "from sklearn.model_selection import train_test_split"?

#dir(df_train)
df_tgt=df.ACCEPTED

df = df.drop(['ACCEPTED'], axis='columns') # drop target column from dataset
df = df.drop(['ZIP3'], axis='columns') # temporary -> ask to get this column removed from output csv files

#oversample = SMOTE()
#df, df_tgt = oversample.fit_resample(df, df_tgt)

X_train, X_test, y_train, y_test = train_test_split(df, df_tgt, test_size=0.2) # 80% test and 20% training

# parameter changing to be done here
# fixme: get progress to show in some way upon execution
# can take about 15 minutes to run

model = SVC(kernel = 'linear', verbose=True, C=10) #decision_function_shape='ovo', class_weight={0: 606, 1: 2619}, gamma='auto', class_weight={0: 202, 1: 873}
start = time.time()
model.fit(X_train, y_train) #trains model
end = time.time()

print("Model score: " + str(model.score(X_test, y_test))) # tests model accuracy
print("Time taken in seconds: " + str(end - start))

print("PREDICTION")
resultArray = model.predict(X_test)
print(resultArray)
unique, counts = np.unique(resultArray, return_counts=True)
print(dict(zip(unique, counts)))

print(model) # show all of SVM's params