import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score
from os.path import dirname, abspath

path = dirname(dirname(abspath(__file__)))

def KNN():
    train = pd.read_csv(path + '/train_clean.csv')
    test = pd.read_csv(path + '/test_clean.csv')
    number = LabelEncoder()

    for i in train.columns:
        train[i] = number.fit_transform(train[i])

    for i in test.columns:
        test[i] = number.fit_transform(test[i])

    features = [i for i in train.columns[:-1]]
    target = "class"

    features_train, target_train = train[features], train[target]
    features_test, target_test = test[features], test[target]

    model = KNeighborsClassifier()
    model.fit(features_train, target_train)

    pred = model.predict(features_test)
    accuracy = accuracy_score(target_test, pred)
    precision = precision_score(target_test, pred)
    recall = recall_score(target_test, pred)
    return {'accuracy': accuracy * 100, 'precision': precision*100, 'recall': recall*100}