import pandas as pd
from sklearn.model_selection import train_test_split # for splitting dataset
import util

def prepare_data(data_from_path):
    years = [11,12,13,14,15,16,17]
    target = 'ACCEPTED' # the column holding the ground truth value for a student 

    df = pd.read_csv(data_from_path + '10.csv')

    # add more data to dataset
    for y in years:
        df = df.append(pd.read_csv(data_from_path + str(y)+'.csv'))
        
    # dropping all rows that contain an empty value
    df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

    df_tgt=df.ACCEPTED # target dataframe

    df = df.drop([target], axis='columns') # drop target column from dataset
    if "ZIP3" in df.keys():
        df = df.drop(['ZIP3'], axis='columns')

    # scramble the data and split the data 80-20 for training and testing
    X_train, X_test, y_train, y_test = train_test_split(df, df_tgt, test_size=0.2)

    util.save_as_pkl([X_train, X_test, y_train, y_test], "../cleaned_data/data.pkl")

def retrieve_data():
    X_train, X_test, y_train, y_test = util.load_pkl("../cleaned_data/data.pkl")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    prepare_data("../cleaned_data/receivedOffer/receivedOffer_")