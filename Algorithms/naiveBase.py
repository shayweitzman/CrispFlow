import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
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



def ourNaiveBase():
    """
    This function create and evaluate Naive Bayes our implementation.

    :return: results dictionary with: accuracy,precision,recall,majority,TrueNegetive,FalsePositive,FalseNegetive,TruePositive,model keys.
    """
    try:
        df = pd.read_csv(path + '/train_clean.csv')
    except:
        return -1
    #count amount of 'yes' and 'no' values.
    totalNo = (df["class"] == 'no').sum()
    totalYes = (df["class"] == 'yes').sum()
    # the majority between yes/no in the df.
    R = "yes" if totalYes > totalNo else "no"

    def yesNoPercent(df):
        """
        this function creates resultYes and resultNo dictionaries - resultYes: {col1:{val1:probabilyToYes,val2:....},col2:{....}} and same for resultNo.
        :param df: the whole data frame.
        :return: (resultYes,resultNo) as tuple.
        """
        resultYes = {}
        resultNo = {}
        for col in df.columns:
            if col != 'class':
                resultYes[col] = {}
                resultNo[col] = {}
                df1 = pd.crosstab(df[col], df['class'])
                for index, row in df1.iterrows():
                    resultYes[col][index] = row['yes'] / totalYes
                    resultNo[col][index] = row['no'] / totalNo
        return (resultYes, resultNo)

    (resultYes, resultNo) = yesNoPercent(df)
    df = pd.read_csv(path + '/test_clean.csv')

    def getBinPercent(title, value, yesOrNo): # get feature name, specific value and "yes" / "no" , returns the probability according resultYes&resultNo dictionaries.
            try:
                return resultYes[title][value] if yesOrNo == "yes" else resultNo[title][value]
            except:
                return 1.0

    ##### Evaluate the model

    truePositive = trueNegative = falseNegative = falsePositive = 0
    counterYesGuess = counterNoGuess = 0


    #####loop over the test file rows and check our model on each.

    for index, row in df.iterrows():
        sumYes = sumNo = 1
        for title in df.columns:
            if title != "class":
                sumYes *= getBinPercent(title, row[title], "yes")
                sumNo *= getBinPercent(title, row[title], "no")
        sumYes = sumYes * totalYes/(totalYes+totalNo)
        sumNo = sumNo * totalNo/(totalYes+totalNo)

        if sumYes > sumNo:
            counterYesGuess += 1
        else:
            counterNoGuess += 1

        if (sumYes > sumNo and row["class"] == "yes") or (sumYes == sumNo and row["class"] == "yes" and R == 'yes'):
            truePositive += 1
        elif (sumNo > sumYes and row["class"] == "no") or (sumYes == sumNo and row["class"] == "no" and R == 'no'):
            trueNegative += 1
        elif (sumNo > sumYes and row["class"] == "yes") or (sumYes == sumNo and row["class"] == "yes" and R == 'no'):
            falseNegative += 1
        elif (sumYes > sumNo and row["class"] == "no") or (sumYes == sumNo and row["class"] == "no" and R == 'yes'):
            falsePositive +=1

    accuracy = ((trueNegative + truePositive)/(counterYesGuess + counterNoGuess)) * 100 if (counterYesGuess + counterNoGuess) != 0 else 0
    precision = (truePositive/(truePositive + falsePositive))*100 if (truePositive + falsePositive) != 0 else 0
    recall = (truePositive/(truePositive + falseNegative))*100 if (truePositive + falseNegative) != 0 else 0

    return {'accuracy':  accuracy, 'precision': precision, 'recall': recall,'majority':majority(), "TrueNegetive": trueNegative, "FalsePositive": falsePositive, "FalseNegetive": falseNegative, "TruePositive": truePositive, "model": {"Yes": resultYes,"No": resultNo}}

def builtinNaiveBase(discreteFeatures):
    """
    This function create and evaluate Gaussian Naive Bayes from sklearn.
    :param discreteFeatures: dictionary of the categorical features with their unique values.
    :return: results dictionary with: accuracy,precision,recall,majority,TrueNegetive,FalsePositive,FalseNegetive,TruePositive,model keys.
    """
    train = pd.read_csv(path + '/train_clean.csv')
    test = pd.read_csv(path + '/test_clean.csv')
    number = LabelEncoder()

    for i in discreteFeatures:
        train[i] = number.fit_transform(train[i])

    for i in discreteFeatures:
        test[i] = number.fit_transform(test[i])

    features = [i for i in train.columns[:-1]]# all coloums except 'class'

    target = "class"

    features_train, target_train = train[features], train[target]
    features_test, target_test = test[features], test[target]


    model = GaussianNB()
    model.fit(features_train, target_train)

    pred = model.predict(features_test)
    accuracy = accuracy_score(target_test, pred)
    precision = precision_score(target_test, pred)
    recall = recall_score(target_test, pred)
    tn, fp, fn, tp = confusion_matrix(target_test, pred).ravel()

    return {'accuracy': accuracy * 100, 'precision': precision * 100, 'recall': recall * 100,'majority':majority(), "TrueNegetive": tn, "FalsePositive": fp, "FalseNegetive": fn, "TruePositive": tp,"model": model}








