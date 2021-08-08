import math
import itertools
import pandas as pd
import ast

"""
Authors:
Shay Weitzman - 315618918
Yinon Hadad- 315451542
Dolev Peretz - 208901504

"""


######################################### Information Gain , Conditional Entropy , Entropy #########################################
def entropy(data):
    """
    this function gets column and return it entropy.
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
    this funtion get two columns and return their conditional entropy between them.
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
    this function get two columns and return the information gain from them by entropy-(conditional Entropy).
    """
    return entropy(data1) - conditionalEntropy(data1, data2)

def changeN(x, node):
    """this function get value and bin and list contains tuples with bins' borders and check which bin contains the value
    .* createBinDataFrame and changeN are especially for our entropy based entropy.
    """

    for i in range(len(node)):
        if node[i][0] <= x <= node[i][1]:
            return i
    return -1

def createBinDataFrame(df, col, lst):
    """
    this function gets dataframe, specific column and bins , and change the values of each row to bins' value using changeN function
    .* createBinDataFrame and changeN are especially for our entropy based entropy.
    """
    df[col] = df[col].apply(lambda x: changeN(x, lst))
    return df

######################################### Discretizations #########################################

############ Our Implementations ############

def ourCreateBins(discretizationMethod, numOfBining, numericFeatures, df):
    """
    This function creates bins and change df according them.

    :param discretizationMethod: discretization method chose by the user.
    :param numOfBining: number of bins chose by user.
    :param numericFeatures: dictionary of all the numeric features
    :param df: the data frame.
    """
    # Equal width
    if discretizationMethod == 1:
        for i in numericFeatures:
            space = (df[i].max() - df[i].min()) // numOfBining
            space = max(1, space)
            result = []
            for j in range(1, numOfBining):
                result += [(df[i].min()) + space * j]
                numericFeatures[i] = result
    # Equal frequency
    elif discretizationMethod == 2:
        space = len(df) // numOfBining
        df1 = df.copy()
        for i in numericFeatures:
            df1.sort_values(by=[i], inplace=True)
            df1.reset_index(drop=True, inplace=True)
            for j in range(1, numOfBining):
                if 'NUMERIC' in numericFeatures[i]:
                    numericFeatures[i] = [df1[i][space * j]]
                else:
                    numericFeatures[i] += [df1[i][space * j]]
    # entropy Bining
    elif discretizationMethod == 3:
        for col in numericFeatures:
            numericFeatures[col] = ourEntropyBining(col, df.copy(), numOfBining)
    # this steps only for equal width and equal frequency.
    if discretizationMethod != 3:
        for key in numericFeatures:
            if 'NUMERIC' in numericFeatures[key]:
                numericFeatures[key][0] = -math.inf
            else:
                numericFeatures[key] = [-math.inf] + numericFeatures[key]
            numericFeatures[key].append(math.inf)
        for key in numericFeatures:
            bins = []
            for i in range(len(numericFeatures[key][:-1])):
                bins.append((numericFeatures[key][i], numericFeatures[key][i + 1]))
            numericFeatures[key] = bins


def ourEntropyBining(col, df, numOfBining):
    """
    this function calculate all combinations of splitting to bins and returns the split with the most gain.
    :param col: specific column
    :param df: whole data fram
    :param numOfBining: number of binning chose by the user.
    """
    if numOfBining == 1:
        return [(-math.inf, math.inf)]
    x = df[col]
    x = set(x)
    bestSplit = None
    gain = 0
    for comb in itertools.combinations(x, numOfBining-1):
        comb = sorted(comb)
        bins = []
        for i in range(len(comb)):
            if i == 0:
                bins += [(-math.inf, comb[i])]
            if i == len(comb) - 1:
                bins += [(comb[i], math.inf)]
            else:
                bins += [(comb[i], comb[i + 1])]
        df1 = createBinDataFrame(df.copy(), col, bins)
        current = informationGain(df1[col], df1['class'])
        if gain < current:
            bestSplit = bins
            gain = current
    return bestSplit



############ Built-in Implemantations ############


def CreateBins(discretizationMethod, numOfBining, numericFeatures, df):
    """
    This function creates bins and change df according them using buit-in discretization methods.
    :param discretizationMethod: discretization method chose by the user.
    :param numOfBining: number of bins chose by user.
    :param numericFeatures: dictionary of all the numeric features
    :param df: the data frame.
    """
    df1 = df.copy()
    # Implemented Equal Width
    if discretizationMethod == 4:
        for i in numericFeatures:
            df1[i], numericFeatures[i] = pd.cut(x=df1[i], bins=numOfBining, retbins=True)
    # Implemented Equal Length
    elif discretizationMethod == 5:
        for i in numericFeatures:
            df1[i], numericFeatures[i] = pd.qcut(df[i], q=numOfBining, duplicates='drop', retbins=True)

    for i in numericFeatures:
        numericFeatures[i][0] = -math.inf
        numericFeatures[i][len(numericFeatures[i]) - 1] = math.inf

    for key in numericFeatures:
        b = []
        for i in range(len(numericFeatures[key][:-1])):
            b.append((numericFeatures[key][i], numericFeatures[key][i + 1]))
        numericFeatures[key] = b



### Applying Bins on the data files.

def applyBinsOnDF(df, numericFeatures):
    """
     this function gets dataframe, and numeric features dictionary with the bins , and change the values of each row to bins' value using changeValuesToBin function.
     """
    for i in numericFeatures:
        df[i] = df[i].apply(lambda x: changeValuesToBin(x, numericFeatures[i]))
    return df

def changeValuesToBin(x, node):
    """
    this function get value and list contains tuples with bins' borders and check which bin contains the value
    """

    for i in range(len(node)):
        if node[i][0] == x == node[i][1]:
            return i
        if node[i][0] <= x < node[i][1]:
            return i
    return -1
