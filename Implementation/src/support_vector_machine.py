import numpy as np
import pandas as pd
import time # for timing
from sklearn.svm import SVC # actual support vector classifier
from sklearn.model_selection import train_test_split # for splitting dataset
import util
import data

def retrieve_model():
    model = util.load_pkl("../models/SVM.pkl")
    return model

def create_model():
    X_train, _, y_train, _ = data.retrieve_data()

    # define and run SVM model
    model = SVC(kernel = 'linear', verbose=False, C=10)
    model.fit(X_train, y_train) #trains model

    util.save_as_pkl(model, "../models/SVM.pkl")
    return model

if __name__ == "__main__":
    create_model()