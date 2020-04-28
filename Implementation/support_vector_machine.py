import numpy
import time # for timing training of data
import pandas as pd
from sklearn.svm import SVC # actual support vector classifier
from sklearn.model_selection import train_test_split # for splitting dataset

# load cleaned up csvs into dataframe
path = '../cleaned_data/receivedOffer/receivedOffer_' #allStudents/allStudents_'
years = [11,12,13,14,15,16] # 17 <- used for testing created model
target = 'ACCEPTED'

df_test = pd.read_csv(path + '17.csv')
df_train = pd.read_csv(path + '10.csv')

for y in years:
    #print(' + Year: ' + str(y) + ' = ')
    df_train = df_train.append(pd.read_csv(path + str(y)+'.csv'))
    
print('Total students read in: ' + str(len(df_train)))

df_train.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True) #dropping all rows that contain an empty value

print('Total students ready for ML alg: ' + str(len(df_train)))

#maybe combining and scrambling the data and splitting the data 80-20 for training and testing would be better?
#so "from sklearn.model_selection import train_test_split"?

#dir(df_train)
df_train_tgt=df_train.ACCEPTED
df_test_tgt=df_test.ACCEPTED

# drop target columns from datasets
#df_test.head()
df_test = df_test.drop([target], axis='columns')
df_test = df_test.drop(['ZIP3'], axis='columns') # temporary -> ask to get this column removed from output csv files
df_train = df_train.drop(['ZIP3'], axis='columns') # temporary -> ask to get this column removed from output csv files
df_train = df_train.drop([target], axis='columns')
#df_test.head()

X_train, X_test, y_train, y_test = train_test_split(df_train, df_train_tgt, test_size=0.2) # 80% test and 20% training

# parameter changing to be done here
# fixme: get progress to show in some way upon execution
# can take about 15 minutes to run

model = SVC(verbose=True, gamma=1000, C=10) #decision_function_shape='ovo', class_weight={0: 606, 1: 2619}, gamma='auto', class_weight={0: 202, 1: 873}
start = time.time()
model.fit(X_train, y_train) #get 1d array as the target
end = time.time()

print("Model score: " + str(model.score(X_test, y_test)))
print("Time taken in seconds: " + str(end - start))

print(model) # show all of SVM's params