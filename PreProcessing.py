import numpy as np
import pandas as pd
from tkinter import messagebox as mb
import math
import Discretization
import entropy_based_binning as ebb

"""
Authors:
Shay Weitzman - 315618918
Yinon Hadad- 315451542
Dolev Peretz - 208901504

"""

discreteFeatures = {}
numericFeatures = {}

def cleanData(TrainFile, TestFile, StructFile, fillEmptyMethod, normalizationVal, discretizationMethod, binsSpinBox):
    """
    :param TrainFile: the path to the train file
    :param TestFile: the path to the test file
    :param StructFile: the path to the Struct file
    :param fillEmptyMethod: the chosen way to fill empty values
    :param normalizationVal: the chosen normalization
    :param discretizationMethod: the chosen discretization method
    :param binsSpinBox: the bins number
    :return: True if the cleaning file successfully created else return false
    """
    global discreteFeatures
    global numericFeatures
    files = {'train': pd.read_csv(TrainFile), 'test': pd.read_csv(TestFile)}

   # create dictionary for all the columns and their unique values for categorical cols and 'NUMERIC' for numeric features.
    for key in files:
        if files[key].empty:
            mb.showerror("Error", "One of the files is empty")
            return False
        try:  # open all the files
            with open(StructFile, "r") as f:
                discreteFeatures = {}

                for line in f:
                    line = line.split()
                    discreteFeatures[line[1]] = line[2].replace("{", "").replace("}", "").split(',')
        except:  # problem with the files
            mb.showerror("Error", "There is problem with the files")
            return False
        for i in discreteFeatures:
            for j in range(len(discreteFeatures[i])):
                if discreteFeatures[i][j] != 'NUMERIC':
                    discreteFeatures[i][j] = discreteFeatures[i][j].lower()

        numericFeatures = {}
        # Create numeric features' dict.
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
    # Create bins

    # 0 < discretizationMethod < 4 ==> our implementations
    if 0 < discretizationMethod.get() < 4:
        Discretization.ourCreateBins(discretizationMethod.get(), int(binsSpinBox.get()), numericFeatures, files['train'])

    # 4 <= discretizationMethod < 6 ==> builtin implementations
    elif 4 <= discretizationMethod.get() < 6:
        Discretization.CreateBins(discretizationMethod.get(), int(binsSpinBox.get()), numericFeatures, files['train'])
    try:
        # builtin entropy-based discretization.
        if discretizationMethod.get() == 6:
            for i in numericFeatures:
                files['train'][i] = ebb.bin_array(files['train'][i], nbins=int(binsSpinBox.get()), axis=0)
                files['test'][i] = ebb.bin_array(files['test'][i], nbins=int(binsSpinBox.get()), axis=0)
                numericFeatures[i] = [i for i in range(int(binsSpinBox.get()))]
            files['train'].to_csv('train_clean.csv', sep=',', index=False)
            files['test'].to_csv('test_clean.csv', sep=',', index=False)
            return True
    except:
        mb.showerror("Error", "Entropy based bining failed")
        return False

    # change all dataframe values according the bins.
    if discretizationMethod.get() != 0:
        files['train'] = Discretization.applyBinsOnDF(files['train'], numericFeatures)
        files['test'] = Discretization.applyBinsOnDF(files['test'], numericFeatures)
    # create cleaned files after fill NaNs , and binnings.
    try:
        files['train'].to_csv('train_clean.csv', sep=',', index=False)
        files['test'].to_csv('test_clean.csv', sep=',', index=False)
    except:
        mb.showerror("Error", "Close the clean files and try again")
        return False
    return True
