import pandas as pd
import numpy as np
import math
from os.path import dirname, abspath

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from Plot import majority

"""
Authors:
Shay Weitzman - 315618918
Yinon Hadad- 315451542
Dolev Peretz - 208901504

"""


path = dirname(dirname(abspath(__file__)))

class Tree:
    """
    This is the decision tree class, the class contain the name of the column that
    we choose to split the data frame by (the one how give us the max gain),
    and dict of all the bins how belong to this column.
    the value is sub tree till we get to the leaf, there the value is "yes"
    or "on" depends on the majority in the splited data frame
    """
    def __init__(self, divBy):
        """
        :param divBy: the col how chosen to split the data frame
        the builder initialise empty dict for the childrens
        """
        self.divBy = divBy
        self.childrens = {}

    def addChildrens(self, key, val):
        """
        :param key: the value of the bin
        :param val: the sub tree for this bin
        :return:
        """
        self.childrens[key] = val

########### getters #############

    def getDivBy(self):
        return self.divBy

    def getChiled(self, x):
        return self.childrens[x]

def ourDecisionTree(numericFeatures, features, DfLen, Depth, window):
    """
    :param numericFeatures: all the features with the numeric values (there names)
    :param features: all the desecrate features (categorial values)
    :param DfLen: the threshold of the decision tree (len of the data frame)
    :param Depth: the threshold of the decision tree (depth of the tree)
    :param window: as long as the window still display
    :return: results dictionary with: accuracy,precision,recall,majority,TrueNegetive,FalsePositive,FalseNegetive,TruePositive,model keys.
    """
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
        bins[i] = [i for i in range(len(numericFeatures[i]))]
    train = pd.read_csv(path + '/train_clean.csv')
    test = pd.read_csv(path + '/test_clean.csv')

    def findR(df):
        """
        :param df: data frame
        :return: "yes" if there is more yes then no in the data feame else "no"
        """
        n = (df["class"] == 'no').sum()# no classified
        y = (df["class"] == 'yes').sum()# yes classified
        return "yes" if y > n else "no"


    def entropy(data):
        """
        :param data: one domination array
        :return: the value of the entropy of the array
        """
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
        """
        :param data1: one domination array
        :param data2: one domination array
        :return: the value of the conditional entropy between the two arrays
        """
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
        """
        :param data1: one domination array
        :param data2: one domination array
        :return: the information gain value between the two arrays
        """
        return entropy(data1) - conditionalEntropy(data1, data2)

    def rateCol(df):
        """
        :param df: data frame
        :return: The functon callculate and return the column name with the most gain (in front of the class column)
                 and the gain value

        """
        maxGain = 0
        col = ''
        for i in df.columns:
            if i != "class":
                val = informationGain(df[i], df['class'])
                if maxGain < val:
                    maxGain = val
                    col = i
        return maxGain, col



    def buildTree(df, bins, count=Depth):  # new build tree
        """
        This is recursive function, the function build the decision tree
        :param df: splited data frame
        :param bins: the bins of etch column
        :param count: the max depth of the tree
        :return: tree as long as we didnt get to the threshold, else yes or no (depends of the majority in the splited data frame)
        """
        val, col = rateCol(df)
        if count == 0 or len(df) < DfLen or col == '':
            return findR(df)
        tree = Tree(col)
        for i in bins[col]:
            # if col in numericFeatures:
            #     tree.addChildrens(bins[col].index(i), buildTree(df.query(col + ' == ' + str(bins[col].index(i))).drop(col, axis='columns'), bins, count - 1))
            # else:
            tree.addChildrens(i, buildTree(df.query(col + ' == \'' + str(i) + '\'').drop(col, axis='columns'), bins, count - 1))
        return tree

    resultTree = buildTree(train, bins)# the decision tree for the given data frame



    truePositive = trueNegative = falseNegative = falsePositive = 0
    counterYesGuess = counterNoGuess = 0

    for index, row in test.iterrows():
        """
        This part iterate on every row in the test fail and for each row 
        dive to the decision tree till the chilled value is string (yes or no)
        then we check our guess and count the nedded information 
        """
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

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'majority': majority(),"TrueNegetive": trueNegative, "FalsePositive": falsePositive, "FalseNegetive": falseNegative,"TruePositive": truePositive, "model": resultTree}


def builtinDecisionTree(discreteFeatures, DfLen, Depth, window):
    """
    :param discreteFeatures: all the features with the numeric values (there names)
    :param DfLen:  the threshold of the decision tree (len of the data frame)
    :param Depth: the threshold of the decision tree (depth of the tree)
    :param window: as long as the window still display
    :return: results dictionary with: accuracy,precision,recall,majority,TrueNegetive,FalsePositive,FalseNegetive,TruePositive,model keys.

    The function use sklearn library to build the function tree and test the model
    The function return dict of all the information of the decision tree experiment
    """
    try:
        if 'normal' != window.state():
            return
    except:
        return
    train = pd.read_csv(path + '/train_clean.csv')
    test = pd.read_csv(path + '/test_clean.csv')

    number = LabelEncoder()
    for i in discreteFeatures:
        train[i] = number.fit_transform(train[i])

    number = LabelEncoder()
    for i in discreteFeatures:
        test[i] = number.fit_transform(test[i])

    features = [i for i in train.columns[:-1]]
    target = "class"

    features_train, target_train = train[features], train[target]
    features_test, target_test = test[features], test[target]

    model = DecisionTreeClassifier(criterion='entropy', max_depth=Depth, min_samples_split=DfLen)
    model.fit(features_train, target_train)

    pred = model.predict(features_test)
    accuracy = accuracy_score(target_test, pred)
    precision = precision_score(target_test, pred)
    recall = recall_score(target_test, pred)

    tn, fp, fn, tp = confusion_matrix(target_test, pred).ravel()

    return {'accuracy': accuracy * 100, 'precision': precision * 100, 'recall': recall * 100, 'majority': majority(),"TrueNegetive": tn, "FalsePositive": fp, "FalseNegetive": fn, "TruePositive": tp, "model": model}
