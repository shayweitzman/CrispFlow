import numpy as np
import pandas as pd
from tkinter import messagebox as mb
import math
import Discretization

discreteFeatures = {}
numericFeatures = {}

def cleanData(TrainFile, TestFile, StructFile, fillEmptyMethod, normalizationVal, discretizationMethod, binsSpinBox):
    global discreteFeatures
    global numericFeatures
    files = {'train': pd.read_csv(TrainFile), 'test': pd.read_csv(TestFile)}

    """creating the clean files"""
    for key in files:
        try:  # open all the files
            with open(StructFile, "r") as f:
                discreteFeatures = {}

                for line in f:
                    line = line.split()
                    discreteFeatures[line[1]] = line[2].replace("{", "").replace("}", "").split(',')
        except:  # problem with the files
            mb.showerror("Error", "There is problem with the files")
            return False

        numericFeatures = {}
        # Create numeric features' dict
        for key1 in discreteFeatures:
            if discreteFeatures[key1] == ['NUMERIC']:
                numericFeatures[key1] = discreteFeatures[key1]
        #Delete numeric features from discrete features' dict
        for key1 in numericFeatures:
            discreteFeatures.pop(key1)

        for col in discreteFeatures:  # replace all the letters to low
            files[key][col] = files[key][col].apply(lambda x: x.lower() if type(x) == str else x)
        classification = files[key]['class']
        files[key] = files[key].query("@classification == 'yes' or @classification == 'no'")  # delete empty, query only filled yes or no

        completeValues = {}
        for i in numericFeatures:  # fill empty with mean
            completeValues[i] = files[key][i].mean()

        for i in numericFeatures:
            files[key][i] = files[key][i].fillna(completeValues[i])

        if (fillEmptyMethod == 0): # fill according all the data.
            for i in discreteFeatures:  # fill empty with the most commen of all data
                completeValues[i] = files[key][i].value_counts().idxmax()
            for i in discreteFeatures:
                files[key][i] = files[key][i].fillna(completeValues[i])
        else: # fill according same classification only.
            for i in discreteFeatures:  #create completeValues dict : {feature+'yes/no': most-common-value} ,fill empty with the most common of the yes only
                completeValues[i + ' yes'] = files[key].query("@classification == 'yes'")[i].value_counts().idxmax()
                completeValues[i + ' no'] = files[key].query("@classification == 'no'")[i].value_counts().idxmax()

            for i in discreteFeatures: #Mark missing values as NaN
                files[key][i] = files[key][i].fillna(np.nan)

            for i in discreteFeatures: #Change NaNs to the most common value of the column.
                for index, row in files[key].loc[lambda df: df['class'] == 'yes'].loc[lambda df: df[i].isnull()].iterrows():   # iterate only empty and 'yes' classed.
                    files[key].at[index, i] = completeValues[i + ' yes']
                for index, row in files[key].loc[lambda df: df['class'] == 'no'].loc[lambda df: df[i].isnull()].iterrows():
                    files[key].at[index, i] = completeValues[i + ' no']
    # normalization
        if normalizationVal.get() == 1 and discretizationMethod.get() == 0: # enable normalization when discretization off.
            for col in numericFeatures:
                files[key][col] = (files[key][col] - files[key][col].mean()) / files[key][col].std() # Z-Score Normalization
    # Create bining
    if discretizationMethod.get() != 0:
        bins = Discretization.CreateBins(discretizationMethod, int(binsSpinBox.get()), numericFeatures, files['train'])
        files['train'] = Discretization.applyBinsOnDF(files['train'], bins)
        files['test'] = Discretization.applyBinsOnDF(files['test'], bins)

    try:
        files['train'].to_csv('train_clean.csv', sep=',', index=False)
        files['test'].to_csv('test_clean.csv', sep=',', index=False)
    except:
        mb.showerror("Error", "Close the clean files and try again")
        return False
    return True

# def creareFeatureBining(discretizationVal,numOfBining, nomericFeature, df):
#     # Equal width
#     if discretizationVal.get() == 1:
#         for i in nomericFeature:
#             space = (df[i].max()-df[i].min())//numOfBining
#             space = max(1, space)
#             result = []
#             for j in range(numOfBining):
#                 result += [(df[i].min()) + space*j]
#                 nomericFeature[i] = result
#     elif discretizationVal.get() == 2:
#     # Equal frequency
#         space = len(df) // numOfBining
#         df1 = df.copy()
#         for i in nomericFeature:
#             df1.sort_values(by=[i], inplace=True)
#             df1.reset_index(drop=True, inplace=True)
#             nomericFeature[i] = [df1[i][j] for j in range(0, len(df1), space)]
#
#     for key in nomericFeature:
#         nomericFeature[key][0] = -math.inf
#         nomericFeature[key].append(math.inf)
#     for key in nomericFeature:
#         bins = []
#         for i in range(len(nomericFeature[key][:-1])):
#             bins.append((nomericFeature[key][i]+1, nomericFeature[key][i + 1]))
#         nomericFeature[key] = bins
#     print(nomericFeature)
#     return nomericFeature
#
# def changeN(x, node):
#     x = round(x)
#     for i in range(len(node)):
#         if node[i][0] <= x <= node[i][1]:
#             return i
#     return -1
#
# def createBinDataFrame(df, nomericFeature):
#     for i in nomericFeature:
#         df[i] = df[i].apply(lambda x: changeN(x, nomericFeature[i]))
#     return df