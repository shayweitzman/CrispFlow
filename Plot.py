import tkinter as tk
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter
import pandas as pd
from tkinter import ttk
from prettytable import PrettyTable
from tkinter import *
import pickle
from tkinter import messagebox as mb

"""
Authors:
Shay Weitzman - 315618918
Yinon Hadad- 315451542
Dolev Peretz - 208901504

"""

def majority():
    """
    This function return the majority rule success percentage.
    """
    train = pd.read_csv('train_clean.csv')
    test = pd.read_csv('test_clean.csv')
    n = (train["class"] == 'no').sum()  # no classified
    y = (train["class"] == 'yes').sum()  # yes classified
    R= 'no' if n>y else 'yes'
    maj = (test['class'] == R).sum()
    return (maj / len(test))*100

def createPikcle(model):
    """
    this function get model as parameter and creates pickle of it.
    :return: none
    """
    try:
        pickle.dump(model, open("Model", 'wb'))
        mb.showinfo("Success", "File named 'Model' created")

    except:
        mb.showerror("Error", "Failed to create pickle")




def showGraph(result):
    """
    this fuction plot graph according to the results of the model.
    :param result: results dictionary with: accuracy,precision,recall,majority,TrueNegetive,FalsePositive,FalseNegetive,TruePositive,model keys.
    :return: none
    """
    root = tk.Tk()
    root.title("Dashboard")
    data1 = {'values': ['Majority', 'Accuracy', 'Precision', 'Recall'],
             'precentage': [majority(), result['accuracy'], result['precision'], result['recall']]
    }

    df1 = DataFrame(data1, columns=['values', 'precentage'])
    figure1 = plt.Figure(figsize=(6, 5), dpi=100)
    ax1 = figure1.add_subplot(111)
    bar1 = FigureCanvasTkAgg(figure1, root)
    bar1.get_tk_widget().grid(column=0, row=0)
    df1 = df1[['values', 'precentage']].groupby('values').sum()
    df1.plot(kind='bar', legend=False, ax=ax1, rot=0, xlabel='')
    ax1.set_title('Majority: {} %, Accuracy: {} %, Precision: {} %, Recall: {} %'.format(int(data1['precentage'][0]), int(result['accuracy']), int(result['precision']), int(result['recall'])))

    nextB = tkinter.Button(root, text='Create Pickle', font=('calibre', 13, 'bold'), command=lambda: createPikcle(result["model"]), padx=25, pady=10)
    nextB.grid(column=0, row=1)

    t = Text(root)

    x = PrettyTable()

    x.field_names = ["", "Negative", "Positive"]

    x.add_row(["No", result["TrueNegetive"], result["FalsePositive"]])
    x.add_row(["Yes", result["FalseNegetive"], result["TruePositive"]])

    t.tag_configure("center", justify='center')
    t.insert(INSERT, x)  # Inserting table in text widget
    t.grid(column=1, row=0)
    t.config(state=DISABLED)
    root.configure(background='white')

    root.mainloop()