import tkinter
from tkinter import ttk
from tkinter import filedialog
import os
from pathlib import Path
import csv
import plt1
from tkinter import messagebox as mb

from Algorithms import DecisionTree, naiveBase, KNN, Kmeans
import PreProcessing
from PreProcessing import cleanData

TrainFile = None
TestFile = None
StructFile = None

Depth = 20
DfLen = 100


def openDir(obj, structureL, trainL, testL, nextB):
    global TrainFile
    global TestFile
    global StructFile
    TrainFile = None
    TestFile = None
    StructFile = None

    obj.filename = filedialog.askdirectory()
    if obj.filename == '':
        return
    files = os.listdir(obj.filename)
    for f in files:
        if ("Structure.txt" in f):
            StructFile = df = obj.filename + '/' + f
        elif ("train.csv" in f):
            TrainFile = obj.filename + '/' + f
        elif ("test.csv" in f):
            TestFile = obj.filename + '/' + f

    if TrainFile != None:
        trainL.configure(text='[V] Train File        ', foreground='green')
    else:
        trainL.configure(text='[X] Train File        ', foreground='red')
    if TestFile != None:
        testL.configure(text='[V] Test File          ', foreground='green')
    else:
        testL.configure(text='[X] Test File          ', foreground='red')
    if StructFile != None:
        structureL.configure(text='[V] Structure File  ', foreground='green')
    else:
        structureL.configure(text='[X] Structure File  ', foreground='red')
    if TrainFile != None and TestFile != None and StructFile != None:
        nextB.configure(state='normal')
    else:
        nextB.configure(state='disable')


######## Next ##########
def nextTab(index):
    if (index == 1):
        if (not cleanData(TrainFile, TestFile, StructFile, fillEmptyMethod, normalizationVal, discretizationMethod, binsSpinBox)):
            return
    tab_control.tab(index, state="disabled")
    tab_control.tab(index + 1, state="normal")
    tab_control.select(index + 1)


######## Back ##########
def backTab(index):
    tab_control.tab(index, state="disabled")
    tab_control.tab(index - 1, state="normal")
    tab_control.select(index - 1)


def switch(bins, val):  # Different algorithm options if Discretization happend or not.
    if val > 0:  # Discretization happend --> Naive Bayse , Decision Tree available.
        bins.configure(state='readonly')
        naiveBase1R.configure(state='normal')
        naiveBase2R.configure(state='normal')
        decisionTree1R.configure(state='normal')
        decisionTree2R.configure(state='normal')
        algorithmVal.set(0)
    else:  # Discretization didn't happend --> Naive Bayse , Decision Tree unavailable.
        bins.configure(state='disabled')
        naiveBase1R.configure(state='disabled')
        naiveBase2R.configure(state='disabled')
        decisionTree1R.configure(state='disabled')
        decisionTree2R.configure(state='disabled')
        algorithmVal.set(4)


def writeResultFile(x):
    done = False
    accuracy = x['accuracy']
    recall = x['recall']
    precision = x['precision']
    discretization = 'Y' if discretizationMethod.get() != 0 else 'N'
    normalization = 'N' if normalizationVal.get() == 0 or discretizationMethod.get() != 0 else 'Y'
    algorithm = 'Our Naive Bayse' if algorithmVal.get() == 0 else 'Implemented Naive Bayse' if algorithmVal.get() == 1 else 'Our Decision Tree' if algorithmVal.get() == 2 else 'Implemented Decision Tree' if algorithmVal.get() == 3 else 'KNN' if algorithmVal.get() == 4 else 'K-Means'
    numOfBining = 'None' if discretizationMethod.get() == 0 else binsSpinBox.get()
    discretizationType = 'Width' if discretizationMethod.get() == 1 else 'Frequency' if discretizationMethod.get() == 2 else 'Entropy Based' if discretizationMethod.get() == 3 else 'None'
    completedBy = 'All Data' if fillEmptyMethod.get() == 0 else 'Classification Column'
    depth = Depth if algorithm == 'Our Decision Tree' else 'Not Relevant'
    dflen = DfLen if algorithm == 'Our Decision Tree' else 'Not Relevant'
    result = [algorithm, discretization, discretizationType, numOfBining, normalization, completedBy,depth,dflen, accuracy, recall, precision]
    result_file = Path('results.csv')

    while not done:
        try:
            if result_file.is_file():  # file already exists
                with open('results.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(result)
            else:  # file just created
                with open('results.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        ['Algorithm', 'Discretization', 'Discretization Type', 'Number Of Bins', 'Normalization',
                         'Completed By', 'Max Tree Depth', 'Min Rows To Split', 'Accuracy', 'Recall', 'Precision'])
                    writer.writerow(result)
            done = True
        except:
            mb.showerror("Error", "Close the results file and try again!")

def choseTrashHold():

    def valuecheck(value, lst, slider):
        newvalue = min(lst, key=lambda x: abs(x - float(value)))
        slider.set(newvalue)

    def chose(root):
        global Depth
        global DfLen
        Depth = slider2.get()
        DfLen = slider1.get()
        root.quit()
        root.destroy()


    root = tkinter.Tk()
    root.geometry("300x200")
    root.title("Threshold Settings")

    tkinter.Label(root, text='').pack()

    tkinter.Label(root, text='Minimum rows to continue tree building?').pack()
    valuelist1 = [i for i in range(10, 101, 10)]
    slider1 = tkinter.Scale(root, from_=min(valuelist1), to=max(valuelist1), variable=20, command=lambda x: valuecheck(x, valuelist1,slider1), orient="horizontal")
    slider1.set(100)
    slider1.pack()

    tkinter.Label(root, text='Max tree depth?').pack()
    valuelist2 = [i for i in range(10, 31)]
    slider2 = tkinter.Scale(root, from_=min(valuelist2), to=max(valuelist2), command=lambda x: valuecheck(x, valuelist2, slider2), orient="horizontal")
    slider2.set(20)
    slider2.pack()

    tkinter.Button(root, text="GO", font=('calibre', 13, 'bold'), padx=25, pady=10, command=lambda: chose(root)).pack()
    root.mainloop()


def algorithmCall():  # Evaluate the model.

    results = {}
    if algorithmVal.get() == 0:
        results = naiveBase.ourNaiveBase(PreProcessing.numericFeatures, PreProcessing.discreteFeatures)
    elif algorithmVal.get() == 1:
        results = naiveBase.builtinNaiveBase()
    elif algorithmVal.get() == 2:
        choseTrashHold()
        results = DecisionTree.ourDecisionTree(PreProcessing.numericFeatures, PreProcessing.discreteFeatures, DfLen, Depth, window)
    elif algorithmVal.get() == 3:
        results = DecisionTree.builtinDecisionTree()
    elif algorithmVal.get() == 4:
        results = KNN.KNN()
    elif algorithmVal.get() == 5:
        results = Kmeans.Kmeans()
    if results != None:
        writeResultFile(results)
        plt1.showGraph(results)




window = tkinter.Tk()
window.geometry('500x300')
# window.iconbitmap("C:\Users\yinon\Downloads\New Project.ico")
window.title("Data Mining")
style = ttk.Style(window)
style.configure('TNotebook.Tab', width=300)
tab_control = ttk.Notebook(window)

################################# Tab 1 ################################################

tab1 = ttk.Frame(tab_control)
tkinter.Label(tab1, text="\t\t", padx=20, pady=10).grid(column=0, row=0)
tab_control.add(tab1, text='Files')

tab1.grid_columnconfigure((0, 1, 2), weight=1)
ttk.Label(tab1, text='Please Choose Files Directory', font=('Helvetica', 13, 'bold')).grid(column=1, row=0)
ttk.Label(tab1, text='', font=('Helvetica', 13, 'bold')).grid(column=1, row=1)

tkinter.Label(tab1, text="\t", font=('calibre', 5)).grid(column=0, row=2)
tkinter.Button(tab1, text="Browse Directory", font=('calibre', 13), command=lambda: openDir(window, structureL, trainL, testL, nextB), padx=20, pady=10).grid(column=1, row=3)

tkinter.Label(tab1, text='').grid(column=1, row=4)
structureL = tkinter.Label(tab1, text='[X] Structure File  ', foreground='red', font=('calibre', 13))
trainL = tkinter.Label(tab1, text='[X] Train File        ', foreground='red', font=('calibre', 13))
testL = tkinter.Label(tab1, text='[X] Test File          ', foreground='red', font=('calibre', 13))
structureL.grid(column=1, row=5)
trainL.grid(column=1, row=6)
testL.grid(column=1, row=7)

nextB = tkinter.Button(tab1, text='Next', font=('calibre', 13, 'bold'), command=lambda: nextTab(0), padx=25, pady=10,
                       state="disabled")
nextB.grid(column=3, row=9)

################################# Tab 2 ################################################

tab2 = ttk.Frame(tab_control)
tab_control.add(tab2, text='Pre-Process')
tab_control.tab(1, state="disabled")
tab2.grid_columnconfigure((0, 1, 2), weight=1)

""" This part responsible for the completation of missing values methood"""

tkinter.Label(tab2, text='\nComplete Missing Values', font=('Helvetica', 9, 'bold')).grid(column=0, row=0, sticky='w')
#tkinter.Label(tab2, text='According To:', font=('Helvetica', 9, 'bold')).grid(column=0, row=1, sticky='w')
# tkinter.Label(tab2, text='in relation to:\n', font=('Helvetica', 9, 'bold')).grid(column=0, row=2, sticky='w')

fillEmptyMethod = tkinter.IntVar()  # 0 its all data, 1 its just infront of the class column
fillEmptyMethod.set(0)
tkinter.Radiobutton(tab2, text="All Data", variable=fillEmptyMethod, value=0).grid(column=0, row=3, sticky='w')
tkinter.Radiobutton(tab2, text="Classification Column", variable=fillEmptyMethod, value=1).grid(column=0, row=4, sticky='w')

""" normalization needed?"""

tkinter.Label(tab2, text='\nIs normalization necessary?', font=('Helvetica', 9, 'bold')).grid(column=1, row=0, sticky='w')

normalizationVal = tkinter.IntVar()  # 0 its no 1 its yes
normalizationVal.set(1)
tkinter.Radiobutton(tab2, text="Yes", variable=normalizationVal, value=1).grid(column=1, row=3, sticky='w')
tkinter.Radiobutton(tab2, text="No", variable=normalizationVal, value=0).grid(column=1, row=4, sticky='w')

""" discretization needed?"""

tkinter.Label(tab2, text='\nDiscretization Type', font=('Helvetica', 9, 'bold')).grid(column=2, row=0, sticky='w')

tkinter.Label(tab2, text='\nBins Number:', font=('Helvetica', 9, 'bold')).grid(column=2, row=7, sticky='w')  # number of bins
binsSpinBox = tkinter.Spinbox(tab2, from_=1, to=50, state="disabled")
binsSpinBox.grid(column=2, row=8, sticky='w')

discretizationMethod = tkinter.IntVar()
discretizationMethod.set(0)  # 0 its none, 1 equal with and 2 its equal length
tkinter.Radiobutton(tab2, text="None", command=lambda: switch(binsSpinBox, discretizationMethod.get()), variable=discretizationMethod, value=0).grid(column=2, row=3, sticky='w')
tkinter.Radiobutton(tab2, text="Equal Width", command=lambda: switch(binsSpinBox, discretizationMethod.get()), variable=discretizationMethod, value=1).grid(column=2, row=4, sticky='w')
tkinter.Radiobutton(tab2, text="Equal Length/Frequency", command=lambda: switch(binsSpinBox, discretizationMethod.get()), variable=discretizationMethod, value=2).grid(column=2, row=5, sticky='w')
tkinter.Radiobutton(tab2, text="By Entropy", command=lambda: switch(binsSpinBox, discretizationMethod.get()), variable=discretizationMethod, value=3).grid(column=2, row=6, sticky='w')


tkinter.Label(tab2, text='        ', font=('Helvetica', 15)).grid(column=2, row=9, sticky='w')
tkinter.Label(tab2, text='\t', font=('Helvetica', 10)).grid(column=2, row=9, sticky='w')
tkinter.Button(tab2, text="Next", font=('calibre', 13, 'bold'), command=lambda: nextTab(1), padx=25, pady=10).grid(column=2, row=11, sticky='e')
tkinter.Button(tab2, text="Back", font=('calibre', 13, 'bold'), padx=25, pady=10, command=lambda: backTab(1)).grid(column=0, row=11, sticky='w')

################################# Tab 3 ################################################

tab3 = ttk.Frame(tab_control)
tab_control.add(tab3, text='Algorithms')
tab_control.tab(2, state="disabled")

"""Algorithm"""
ttk.Label(tab3, text="\n\tChoose Algorithm               ", font=('Helvetica', 13, 'bold')).grid(column=1, row=0)

algorithmVal = tkinter.IntVar()
algorithmVal.set(4)
naiveBase1R = tkinter.Radiobutton(tab3, text="Naive Bayes (our implementation)   ", variable=algorithmVal, value=0, state='disable')
naiveBase1R.grid(column=1, row=1)
naiveBase2R = tkinter.Radiobutton(tab3, text="Naive Bayes (existing library)            ", variable=algorithmVal, value=1, state='disable')
naiveBase2R.grid(column=1, row=2)
decisionTree1R = tkinter.Radiobutton(tab3, text="Decision Tree (our implementation)", variable=algorithmVal, value=2, state='disable')
decisionTree1R.grid(column=1, row=3)
decisionTree2R = tkinter.Radiobutton(tab3, text="Decision Tree (existing library)          ", variable=algorithmVal, value=3, state='disable')
decisionTree2R.grid(column=1, row=4)
tkinter.Radiobutton(tab3, text="KNN                                                      ", variable=algorithmVal, value=4).grid(column=1, row=5)
tkinter.Radiobutton(tab3, text="K-MEANS                                             ", variable=algorithmVal, value=5).grid(column=1, row=6)

ttk.Label(tab3, text='\t', font=('Helvetica', 15, 'bold')).grid(column=1, row=7)

tkinter.Button(tab3, text="Back", font=('calibre', 13, 'bold'), padx=25, pady=10, command=lambda: backTab(2)).grid(column=0, row=10, sticky='w')
tkinter.Button(tab3, text=" Go ", font=('calibre', 13, 'bold'), command=lambda: algorithmCall(), padx=25, pady=10).grid(column=2, row=10, sticky='e')


tab_control.pack(expand=1, fill='both')
window.resizable(width=False, height=False)
window.mainloop()



