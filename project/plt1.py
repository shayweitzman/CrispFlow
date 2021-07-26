import tkinter as tk
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def showGraph(result):
    root = tk.Tk()
    root.title("Dashboard")
    data1 = {'values': ['Accuracy', 'Precision', 'Recall'],
             'precentage': [result['accuracy'], result['precision'], result['recall']]
    }

    df1 = DataFrame(data1, columns=['values', 'precentage'])
    figure1 = plt.Figure(figsize=(6, 5), dpi=100)
    ax1 = figure1.add_subplot(111)
    bar1 = FigureCanvasTkAgg(figure1, root)
    bar1.get_tk_widget().grid(column=0, row=0)
    df1 = df1[['values', 'precentage']].groupby('values').sum()
    df1.plot(kind='bar', legend=False, ax=ax1, rot=0, xlabel='')
    ax1.set_title('Accuracy: {} %, Precision: {} %, Recall: {} %'.format(int(result['accuracy']), int(result['precision']), int(result['recall'])))



    #
    #
    # data2 = {'Year': [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010],
    #          'Unemployment_Rate': [9.8, 12, 8, 7.2, 6.9, 7, 6.5, 6.2, 5.5, 6.3]
    #
    # df2 = DataFrame(data2, columns=['Year', 'Unemployment_Rate'])

    # figure2 = plt.Figure(figsize=(5, 4), dpi=80)
    # ax2 = figure2.add_subplot(111)
    # line2 = FigureCanvasTkAgg(figure2, root)
    # line2.get_tk_widget().grid(column=1, row=0)
    # df2 = df2[['Year', 'Unemployment_Rate']].groupby('Year').sum()
    # df2.plot(kind='line', legend=True, ax=ax2, color='r', marker='o', fontsize=10)
    # ax2.set_title('Year Vs. Unemployment Rate')
    #
    # data3 = {'Interest_Rate': [5, 5.5, 6, 5.5, 5.25, 6.5, 7, 8, 7.5, 8.5],
    #          'Stock_Index_Price': [1500, 1520, 1525, 1523, 1515, 1540, 1545, 1560, 1555, 1565]
    #          }
    # df3 = DataFrame(data3, columns=['Interest_Rate', 'Stock_Index_Price'])
    #
    # figure3 = plt.Figure(figsize=(5, 4), dpi=80)
    # ax3 = figure3.add_subplot(111)
    # ax3.scatter(df3['Interest_Rate'], df3['Stock_Index_Price'], color='g')
    # scatter3 = FigureCanvasTkAgg(figure3, root)
    # scatter3.get_tk_widget().grid(column=0, row=1)
    # ax3.legend(['Stock_Index_Price'])
    # ax3.set_xlabel('Interest Rate')
    # ax3.set_title('Interest Rate Vs. Stock Index Price')
    #
    # data4 = {'Interest_Rate': [5, 5.5, 6, 5.5, 5.25, 6.5, 7, 8, 7.5, 8.5],
    #          'Stock_Index_Price': [1500, 1520, 1525, 1523, 1515, 1540, 1545, 1560, 1555, 1565]
    #          }
    # df4 = DataFrame(data3, columns=['Interest_Rate', 'Stock_Index_Price'])
    #
    # figure4 = plt.Figure(figsize=(5, 4), dpi=80)
    # ax4 = figure4.add_subplot(111)
    # ax4.scatter(df4['Interest_Rate'], df4['Stock_Index_Price'], color='r')
    # scatter4 = FigureCanvasTkAgg(figure4, root)
    # scatter4.get_tk_widget().grid(column=1, row=1)
    # ax4.legend(['Stock_Index_Price'])
    # ax4.set_xlabel('Interest Rate')
    # ax4.set_title('Interest Rate Vs. Stock Index Price')

    root.configure(background='white')

    root.mainloop()