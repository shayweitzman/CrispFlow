import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from os.path import dirname, abspath
from Plot import majority

"""
Authors:
Shay Weitzman - 315618918
Yinon Hadad- 315451542
Dolev Peretz - 208901504

"""


path = dirname(dirname(abspath(__file__)))

def KNN(discreteFeatures, numOfNeighbors, window):
    """
    this function create KNN model and evaluate it.
    :param discreteFeatures: dictionary of all categorical values by this form  {feature1:[unique values],.... }
    :param numOfNeighbors: number of neighbors chose by the user.
    :param window: main window in order to prevent errors.
    :return: results dictionary with: accuracy,precision,recall,majority,TrueNegetive,FalsePositive,FalseNegetive,TruePositive,model keys.
    """
    try:
        if 'normal' != window.state(): #return none if main windows closed.
            return
    except:
        return
    train = pd.read_csv(path + '/train_clean.csv')
    test = pd.read_csv(path + '/test_clean.csv')
    number = LabelEncoder()

    for i in discreteFeatures:
        train[i] = number.fit_transform(train[i])

    for i in discreteFeatures:
        test[i] = number.fit_transform(test[i])

    features = [i for i in train.columns[:-1]]
    target = "class"

    features_train, target_train = train[features], train[target]
    features_test, target_test = test[features], test[target]

    model = KNeighborsClassifier(n_neighbors=numOfNeighbors)
    model.fit(features_train, target_train)

    pred = model.predict(features_test)
    accuracy = accuracy_score(target_test, pred)
    precision = precision_score(target_test, pred)
    recall = recall_score(target_test, pred)
    tn, fp, fn, tp = confusion_matrix(target_test, pred).ravel()

    return {'accuracy': accuracy * 100, 'precision': precision*100, 'recall': recall*100,'majority':majority(), "TrueNegetive": tn, "FalsePositive": fp, "FalseNegetive": fn, "TruePositive": tp, "model": model}