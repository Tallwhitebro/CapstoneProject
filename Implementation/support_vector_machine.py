import numpy as np
import pandas as pd
import time # for timing
from sklearn.svm import SVC # actual support vector classifier
from sklearn.model_selection import train_test_split # for splitting dataset
import pickle

# Save object into compressed pickle file
def save_as_pkl(object, path):
	pickle.dump(object, open(path, "wb"))

# load cleaned up csvs into dataframe
path = '../cleaned_data/receivedOffer/receivedOffer_'
years = [11,12,13,14,15,16, 17]
target = 'ACCEPTED'

df = pd.read_csv(path + '10.csv')

# add more data to dataset
for y in years:
    df = df.append(pd.read_csv(path + str(y)+'.csv'))
    
#dropping all rows that contain an empty value
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

#dir(df_train)
df_tgt=df.ACCEPTED

df = df.drop([target], axis='columns') # drop target column from dataset
df = df.drop(['ZIP3'], axis='columns') # temporary -> ask to get this column removed from output csv files

# scramble the data and split the data 80-20 for training and testing
X_train, X_test, y_train, y_test = train_test_split(df, df_tgt, test_size=0.2)

# define and run SVM model
model = SVC(kernel = 'linear', verbose=True, C=10)
start = time.time()
print("Started training...")
model.fit(X_train, y_train) #trains model
print("... done training.")
end = time.time()

# test model accuracy
print("Model score: " + str(model.score(X_test, y_test)))
print("Time taken in seconds: " + str(end - start))

# predicting results
print("PREDICTION: ")
resultArray = model.predict(X_test)
print(resultArray)
unique, counts = np.unique(resultArray, return_counts=True)
print(dict(zip(unique, counts)))

print(model) # show all of SVM's params

save_as_pkl(model, "../models/SVM.pkl")
