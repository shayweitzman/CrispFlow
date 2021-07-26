import math
import itertools
import pandas as pd


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

def changeN(x, node):
    for i in range(len(node)):
        if node[i][0] <= x <= node[i][1]:
            return i
    return -1

def createBinDataFrame(df, col, lst):
    df[col] = df[col].apply(lambda x: changeN(x, lst))
    return df

def entropyBining(col, df, numOfBining):
    #df = df[[col, 'class']]
    x = df[col]
    x = set(x)
    bast = None
    gain = 0
    for comb in itertools.combinations(x, numOfBining-1):
        comb = sorted(comb)
        bins = []
        for i in range(len(comb)):
            if i == 0:
                bins += [(-math.inf, comb[i])]
            if i == len(comb) - 1:
                bins += [(comb[i] + 1, math.inf)]
            else:
                bins += [(comb[i] + 1, comb[i + 1])]
        df1 = createBinDataFrame(df.copy(), col, bins)
        current = informationGain(df1[col], df1['class'])
        if gain < current:
            bast = bins
            gain = current
    return bast

def CreateBins(discretizationMethod, numOfBining, numericFeatures, df):
    # Equal width
    if discretizationMethod.get() == 1:
        for i in numericFeatures:
            space = (df[i].max()-df[i].min())//numOfBining
            space = max(1, space)
            result = []
            for j in range(numOfBining):
                result += [(df[i].min()) + space*j]
                numericFeatures[i] = result
    elif discretizationMethod.get() == 2:
    # Equal frequency
        space = len(df) // numOfBining
        df1 = df.copy()
        for i in numericFeatures:
            df1.sort_values(by=[i], inplace=True)
            df1.reset_index(drop=True, inplace=True)
            numericFeatures[i] = [df1[i][j] for j in range(0, len(df1), space)]
    #entropy Bining
    elif discretizationMethod.get() == 3:
        for col in numericFeatures:
            numericFeatures[col] = entropyBining(col, df.copy(), numOfBining)
            #print(numericFeatures)
    if discretizationMethod.get() != 3:
        for key in numericFeatures:
            numericFeatures[key][0] = -math.inf
            numericFeatures[key].append(math.inf)
        for key in numericFeatures:
            bins = []
            for i in range(len(numericFeatures[key][:-1])):
                bins.append((numericFeatures[key][i] + 1, numericFeatures[key][i + 1]))
            numericFeatures[key] = bins

    return numericFeatures

def changeValuesToBin(x, node):
    x = round(x)
    for i in range(len(node)):
        if node[i][0] <= x <= node[i][1]:
            return i
    return -1

def applyBinsOnDF(df, numericFeatures):
    for i in numericFeatures:
        df[i] = df[i].apply(lambda x: changeValuesToBin(x, numericFeatures[i]))
    return df