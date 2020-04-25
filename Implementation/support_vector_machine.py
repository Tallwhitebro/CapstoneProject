import numpy
import pandas as pd
from sklearn.svm import SVC # actual support vector classifier

# load cleaned up csvs into dataframe
path = '../cleaned_data/allStudents/allStudents_'
years = [11,12,13,14,15,16] # 17 <- used for testing created model
target = 'ACCEPTED'

df_test = pd.read_csv(path + '17.csv')
df_train = pd.read_csv(path + '10.csv')

for y in years:
    #print(' + Year: ' + str(y) + ' = ')
    df_train = df_train.append(pd.read_csv(path + str(y)+'.csv'))
    
print('Total students: ' + str(len(df_train)))

#maybe combining and scrambling the data and splitting the data 80-20 for training and testing would be better?
#so "from sklearn.model_selection import train_test_split"?

#dir(df_train)
df_train_tgt=pd.DataFrame(df_train.ACCEPTED)
df_test_tgt=pd.DataFrame(df_test.ACCEPTED)

# drop target columns from datasets
#df_test.head()
df_test = df_test.drop(['ACCEPTED'], axis='columns')
df_train = df_train.drop(['ACCEPTED'], axis='columns')
#df_test.head()

model = SVC()
model.fit(df_train, df_train_tgt.ACCEPTED.ravel()) #get 1d array as the target

# parameter changing to be done here
# fixme: get progress to show in some way upon execution
# can take about 15 minutes to run

print(model) # show SVM's params

print("Model score: " + str(model.score(df_test, df_test_tgt)))