import pandas as pd
import numpy as np
import math
from os.path import dirname, abspath

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier

path = dirname(dirname(abspath(__file__)))

def ourDecisionTree(numericFeatures, features, DfLen, Depth, window):
    try:
        if 'normal' != window.state():
            return
    except:
        return
    bins = {}
    for i in features:
        if i != 'class':
            bins[i] = features[i]
    for i in numericFeatures:
        bins[i] = numericFeatures[i]
    train = pd.read_csv(path + '/train_clean.csv')
    test = pd.read_csv(path + '/test_clean.csv')

    def findR(df):
        n = (df["class"] == 'no').sum()# no classified
        y = (df["class"] == 'yes').sum()# yes classified
        return "yes" if y > n else "no"


    def entropy(data):
        unique = {}
        for i in data:
            if i not in unique.keys():
                unique[i] = 1
            else:
                unique[i] += 1
        result = 0
        for i in unique:
            p = unique[i] / len(data)
            result += p * (math.log(p, 2))
        return -1 * result

    def conditionalEntropy(data1, data2):
        x1 = {}
        x2 = {}
        for i in data1:
            if i not in x1.keys():
                x1[i] = 1
            else:
                x1[i] += 1
        for i in data2:
            if i not in x2.keys():
                x2[i] = 1
            else:
                x2[i] += 1
        x3 = {}
        for i in x1:
            for j in x2:
                x3[(i, j)] = 0
        for first, sec in zip(data1, data2):
            x3[(first, sec)] += 1

        pMutual = {}
        pConditional = {}
        for i in x3:
            pMutual[i] = x3[i] / len(data1)
        for i in x3:
            pConditional[i] = x3[i] / x2[i[1]]

        result = 0

        for i in pMutual.keys():
            result += pMutual[i] * math.log(pConditional[i], 2) if pConditional[i] != 0 else 0

        return -1 * result

    def informationGain(data1, data2):
        return entropy(data1) - conditionalEntropy(data1, data2)

    def rateCol(df):
        maxGain = 0
        col = ''
        for i in df.columns:
            if i != "class":
                val = informationGain(df[i], df['class'])
                if maxGain < val:
                    maxGain = val
                    col = i
        return maxGain, col

    class Tree:
        def __init__(self, divBy):
            self.divBy = divBy
            self.childrens = {}

        def addChildrens(self, key, val):
            self.childrens[key] = val

        def getDivBy(self):
            return self.divBy

        def getChiled(self, x):
            return self.childrens[x]

    def buildTree(df, bins, count=Depth):  # new build tree
        val, col = rateCol(df)
        if count == 0 or len(df) < DfLen or col == '':
            return findR(df)
        tree = Tree(col)
        for i in bins[col]:
            if col in numericFeatures:
                tree.addChildrens(bins[col].index(i), buildTree(df.query(col + ' == ' + str(bins[col].index(i))).drop(col, axis='columns'), bins, count - 1))
            else:
                tree.addChildrens(i, buildTree(df.query(col + ' == \'' + str(i) + '\'').drop(col, axis='columns'), bins, count - 1))
        return tree
    resultTree = buildTree(train, bins)



    truePositive = trueNegative = falseNegative = falsePositive = 0
    counterYesGuess = counterNoGuess = 0

    for index, row in test.iterrows():
        runTree = resultTree
        while (isinstance(runTree, Tree)):
            runTree = runTree.getChiled(row[runTree.getDivBy()])

        if runTree == 'yes':
            counterYesGuess += 1
        else:
            counterNoGuess += 1

        if row['class'] == 'yes' and runTree == 'yes':
            truePositive += 1
        elif row['class'] == 'no' and runTree == 'no':
            trueNegative += 1
        elif row['class'] == 'yes' and runTree == 'no':
            falseNegative += 1
        elif row['class'] == 'no' and runTree == 'yes':
            falsePositive += 1

    accuracy = ((trueNegative + truePositive) / (counterYesGuess + counterNoGuess)) * 100 if (counterYesGuess + counterNoGuess) != 0 else 0
    precision = (truePositive / (truePositive + falsePositive)) * 100 if (truePositive + falsePositive) != 0 else 0
    recall = (truePositive / (truePositive + falseNegative)) * 100 if (truePositive + falseNegative) != 0 else 0

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall}



def builtinDecisionTree():
    train = pd.read_csv(path + '/train_clean.csv')
    test = pd.read_csv(path + '/test_clean.csv')

    number = LabelEncoder()
    for i in train.columns:
        train[i] = number.fit_transform(train[i])

    number = LabelEncoder()
    for i in test.columns:
        test[i] = number.fit_transform(test[i])

    features = [i for i in train.columns[:-1]]
    target = "class"

    features_train, target_train = train[features], train[target]
    features_test, target_test = test[features], test[target]

    model = DecisionTreeClassifier(criterion='entropy', max_depth=20)
    model.fit(features_train, target_train)

    pred = model.predict(features_test)
    accuracy = accuracy_score(target_test, pred)
    precision = precision_score(target_test, pred)
    recall = recall_score(target_test, pred)
    return {'accuracy': accuracy * 100, 'precision': precision * 100, 'recall': recall * 100}