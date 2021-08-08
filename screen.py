import tkinter
from tkinter import ttk
from tkinter import filedialog
import os
from pathlib import Path
import csv
import Plot
from tkinter import messagebox as mb
from Algorithms import DecisionTree, naiveBase, KNN, Kmeans
import PreProcessing
from PreProcessing import cleanData

"""
Authors:
Shay Weitzman - 315618918
Yinon Hadad- 315451542
Dolev Peretz - 208901504

"""


TrainFile = None
TestFile = None
StructFile = None
secondQ = 20
firstQ = 100


def openDir(obj, structureL, trainL, testL, nextB):
    """
    :param obj: the window
    :param structureL: the red label of the structure file
    :param trainL:the red label of the train file
    :param testL:the red label of the test file
    :param nextB: the button of the next tab
    The function ask the user to enter the directory for the three files,
    if all the files exist in the directory the next button is enabled else
    the button stay disable and the labels show the user witch file missing
    """
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
    """
    :param index: index of the current tab
    the function the function moves the user to the next tab.
    if the user is in the second tab, the function call the clean data function
    """
    if (index == 1):
        if (not cleanData(TrainFile, TestFile, StructFile, fillEmptyMethod, normalizationVal, discretizationMethod, binsSpinBox)):
            return
    tab_control.tab(index, state="disabled")
    tab_control.tab(index + 1, state="normal")
    tab_control.select(index + 1)


######## Back ##########
def backTab(index):
    """
    :param index: index of the current tab
    The function moves the user to the last tab.
    """
    tab_control.tab(index, state="disabled")
    tab_control.tab(index - 1, state="normal")
    tab_control.select(index - 1)


def switch(bins, val):  # Different algorithm options if Discretization happend or not.
    """
    :param bins: the SpinBox of the number of binning
    :param val: discretization type
    The function lick the bins spinBox and the radioButtons of the
    algorithms that require discretization if the user chose without discretization
    """
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


def writeResultFile(modelResults):
    """
    this function get result dictionary and creates/writes to file according to them.
    :param modelResults: dictionary with all the information about the result of the model evaluation.
    """
    done = False
    accuracy = modelResults['accuracy']
    recall = modelResults['recall']
    precision = modelResults['precision']
    majorityRule = modelResults['majority']
    discretization = 'Y' if discretizationMethod.get() != 0 else 'N'
    normalization = 'N' if normalizationVal.get() == 0 or discretizationMethod.get() != 0 else 'Y'
    algorithm = 'Our Naive Bayse' if algorithmVal.get() == 0 else 'Implemented Naive Bayse' if algorithmVal.get() == 1 else 'Our Decision Tree' if algorithmVal.get() == 2 else 'Implemented Decision Tree' if algorithmVal.get() == 3 else 'KNN' if algorithmVal.get() == 4 else 'K-Means'
    numOfBining = 'None' if discretizationMethod.get() == 0 else binsSpinBox.get()
    discretizationType = 'Width(Our)' if discretizationMethod.get() == 1 else 'Frequency(Our)' if discretizationMethod.get() == 2 else 'Entropy Based(Our)' if discretizationMethod.get() == 3 else 'Width(Builtin)' if discretizationMethod.get() == 4 else 'Frequency(Builtin)' if discretizationMethod.get() == 5 else 'Entropy Based(Builtin)' if discretizationMethod.get() == 6 else 'None'
    completedBy = 'All Data' if fillEmptyMethod.get() == 0 else 'Classification Column'
    depth = secondQ if (algorithm == 'Our Decision Tree' or algorithm == 'Implemented Decision Tree')  else 'Not Relevant'
    dflen = firstQ if (algorithm == 'Our Decision Tree' or algorithm == 'Implemented Decision Tree') else 'Not Relevant'
    Neighbors = firstQ if algorithm == 'KNN' else 'Not Relevant'
    maxIterations = firstQ if algorithm == 'K-Means' else 'Not Relevant'
    numOfClusters =  secondQ if algorithm == 'K-Means' else 'Not Relevant'
    result = [algorithm, discretization, discretizationType, numOfBining, normalization, completedBy,depth,dflen,Neighbors,maxIterations,numOfClusters,majorityRule,accuracy, recall, precision]
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
                         'Completed By', 'Max Tree Depth', 'Min Rows To Split','Neighbors','Max Iterations','Number Of Clusters','Majority','Accuracy', 'Recall', 'Precision'])
                    writer.writerow(result)
            done = True
        except:
            mb.showerror("Error", "Close the results file and try again!")


def choseTrashHold(question1, question2, range1, range2, default1, default2, numOfQuestions):
    """
   This function ask the user to enter the threshold for the algorithms how need that
    """

    def valuecheck(value, lst, slider):
        """
        this function get value and slider, check if it valid and set the slider value to it.
        """
        newvalue = min(lst, key=lambda x: abs(x - float(value)))
        slider.set(newvalue)

    def chose(root, num):
        """
        this function set the values of first and second questions of threshold window and close the window.
        """
        global secondQ
        global firstQ
        if num == 2:
            secondQ = slider2.get()
        firstQ = slider1.get()
        root.quit()
        root.destroy()


    root = tkinter.Tk()
    root.geometry("300x200")
    root.title("Threshold Settings")

    tkinter.Label(root, text='').pack()

    tkinter.Label(root, text=question1).pack()
    valuelist1 = [i for i in range1]
    slider1 = tkinter.Scale(root, from_=min(valuelist1), to=max(valuelist1), variable=20, command=lambda x: valuecheck(x, valuelist1,slider1), orient="horizontal")
    slider1.set(default1)
    slider1.pack()
    if numOfQuestions == 2:
        tkinter.Label(root, text=question2).pack()
        valuelist2 = [i for i in range2]
        slider2 = tkinter.Scale(root, from_=min(valuelist2), to=max(valuelist2), command=lambda x: valuecheck(x, valuelist2, slider2), orient="horizontal")
        slider2.set(default2)
        slider2.pack()

    tkinter.Button(root, text="GO", font=('calibre', 13, 'bold'), padx=25, pady=10, command=lambda: chose(root,numOfQuestions)).pack()
    root.mainloop()


def algorithmCall():  # Evaluate the model.

    results = {}
    if algorithmVal.get() == 0:
        results = naiveBase.ourNaiveBase()
    elif algorithmVal.get() == 1:
        results = naiveBase.builtinNaiveBase(PreProcessing.discreteFeatures)
    elif algorithmVal.get() == 2:
        choseTrashHold('Minimum rows to continue tree building?', 'Max tree depth?', range(10, 101, 10), range(1, 31), 100, 20, 2)
        results = DecisionTree.ourDecisionTree(PreProcessing.numericFeatures, PreProcessing.discreteFeatures, firstQ, secondQ, window)
    elif algorithmVal.get() == 3:
        choseTrashHold('Minimum rows to continue tree building?', 'Max tree depth?', range(10, 101, 10), range(10, 31), 100, 20, 2)
        results = DecisionTree.builtinDecisionTree(PreProcessing.discreteFeatures, firstQ, secondQ, window)
    elif algorithmVal.get() == 4:
        choseTrashHold('Number of neighbors?', '', range(1, 21, 1), None, 5, None, 1)
        results = KNN.KNN(PreProcessing.discreteFeatures, firstQ, window)
    elif algorithmVal.get() == 5:
        choseTrashHold('Max iterations?', 'number of cluster?', range(100, 1001, 10), range(2, 11), 300, 8, 2)
        results = Kmeans.Kmeans(PreProcessing.discreteFeatures, firstQ, secondQ, window)
    if results != None:
        writeResultFile(results)
        Plot.showGraph(results)




window = tkinter.Tk()
window.geometry('550x365')
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
ttk.Label(tab1, text='Please Choose Files Directory', font=('Helvetica', 13, 'bold')).grid(column=1, row=1)
ttk.Label(tab1, text='', font=('Helvetica', 17, 'bold')).grid(column=1, row=2)

tkinter.Label(tab1, text="\t", font=('calibre', 5)).grid(column=0, row=3)
tkinter.Button(tab1, text="Browse Directory", font=('calibre', 13), command=lambda: openDir(window, structureL, trainL, testL, nextB), padx=20, pady=10).grid(column=1, row=4)

tkinter.Label(tab1, text='').grid(column=1, row=5)
structureL = tkinter.Label(tab1, text='[X] Structure File  ', foreground='red', font=('calibre', 13))
trainL = tkinter.Label(tab1, text='[X] Train File        ', foreground='red', font=('calibre', 13))
testL = tkinter.Label(tab1, text='[X] Test File          ', foreground='red', font=('calibre', 13))
structureL.grid(column=1, row=6)
trainL.grid(column=1, row=7)
testL.grid(column=1, row=8)

tkinter.Label(tab1, text=" ", font=('calibre', 6, 'bold'), padx=20, pady=10).grid(column=0, row=9)
nextB = tkinter.Button(tab1, text='Next', font=('calibre', 13, 'bold'), command=lambda: nextTab(0), padx=25, pady=10,state="disabled")
nextB.grid(column=3, row=10)

################################# Tab 2 ################################################

tab2 = ttk.Frame(tab_control)
tab_control.add(tab2, text='Pre-Process')
tab_control.tab(1, state="disabled")
tab2.grid_columnconfigure((0, 1, 2), weight=1)

""" This part responsible for the completation of missing values methood"""

tkinter.Label(tab2, text='\nComplete Missing Values', font=('Helvetica', 9, 'bold')).grid(column=0, row=0, sticky='w')

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

#tkinter.Label(tab2, text='\nDiscretization Type', font=('Helvetica', 9, 'bold')).grid(column=0, row=0, sticky='w')

tkinter.Label(tab2, text='\nBins Number:', font=('Helvetica', 9, 'bold')).grid(column=2, row=10, sticky='w')  # number of bins
binsSpinBox = tkinter.Spinbox(tab2, from_=1, to=50, state="disabled")
binsSpinBox.grid(column=2, row=11, sticky='w')

tkinter.Label(tab2, text='\nDiscretization Type', font=('Helvetica', 9, 'bold')).grid(column=2, row=0, sticky='w')

discretizationMethod = tkinter.IntVar()
discretizationMethod.set(0)  # 0 its none, 1 equal with and 2 its equal length
tkinter.Radiobutton(tab2, text="None", command=lambda: switch(binsSpinBox, discretizationMethod.get()), variable=discretizationMethod, value=0).grid(column=2, row=3, sticky='w')
tkinter.Radiobutton(tab2, text="Equal Width (our)", command=lambda: switch(binsSpinBox, discretizationMethod.get()), variable=discretizationMethod, value=1).grid(column=2, row=4, sticky='w')
tkinter.Radiobutton(tab2, text="Equal Length (our)", command=lambda: switch(binsSpinBox, discretizationMethod.get()), variable=discretizationMethod, value=2).grid(column=2, row=5, sticky='w')
tkinter.Radiobutton(tab2, text="By Entropy (our)", command=lambda: switch(binsSpinBox, discretizationMethod.get()), variable=discretizationMethod, value=3).grid(column=2, row=6, sticky='w')
tkinter.Radiobutton(tab2, text="Equal Width", command=lambda: switch(binsSpinBox, discretizationMethod.get()), variable=discretizationMethod, value=4).grid(column=2, row=7, sticky='w')
tkinter.Radiobutton(tab2, text="Equal Length", command=lambda: switch(binsSpinBox, discretizationMethod.get()), variable=discretizationMethod, value=5).grid(column=2, row=8, sticky='w')
tkinter.Radiobutton(tab2, text="By Entropy", fg='red', command=lambda: switch(binsSpinBox, discretizationMethod.get()), variable=discretizationMethod, value=6).grid(column=2, row=9, sticky='w')

tkinter.Label(tab2, text=' ').grid(column=2, row=12, sticky='w')
tkinter.Button(tab2, text="Next", font=('calibre', 13, 'bold'), command=lambda: nextTab(1), padx=25, pady=10).grid(column=2, row=13, sticky='e')
tkinter.Button(tab2, text="Back", font=('calibre', 13, 'bold'), padx=25, pady=10, command=lambda: backTab(1)).grid(column=0, row=13, sticky='w')


################################# Tab 3 ################################################

tab3 = ttk.Frame(tab_control)
tab_control.add(tab3, text='Algorithms')
tab_control.tab(2, state="disabled")

tkinter.Label(tab3, text="\t         ", font=('Helvetica', 9, 'bold'), padx=20, pady=10).grid(column=0, row=0)
tkinter.Label(tab3, text="\t         ", font=('Helvetica', 9, 'bold'), padx=20, pady=10).grid(column=2, row=0)

"""Algorithm"""
ttk.Label(tab3, text="\n\tChoose Algorithm               ", font=('Helvetica', 13, 'bold')).grid(column=1, row=1)

tkinter.Label(tab3, text=" ", font=('Helvetica', 5, 'bold'), padx=20, pady=10).grid(column=2, row=2)

algorithmVal = tkinter.IntVar()
algorithmVal.set(4)
naiveBase1R = tkinter.Radiobutton(tab3, text="Naive Bayes (our implementation)   ", variable=algorithmVal, value=0, state='disable')
naiveBase1R.grid(column=1, row=3)
naiveBase2R = tkinter.Radiobutton(tab3, text="Naive Bayes (existing library)            ", variable=algorithmVal, value=1, state='disable')
naiveBase2R.grid(column=1, row=4)
decisionTree1R = tkinter.Radiobutton(tab3, text="Decision Tree (our implementation)", variable=algorithmVal, value=2, state='disable')
decisionTree1R.grid(column=1, row=5)
decisionTree2R = tkinter.Radiobutton(tab3, text="Decision Tree (existing library)          ", variable=algorithmVal, value=3, state='disable')
decisionTree2R.grid(column=1, row=6)
tkinter.Radiobutton(tab3, text="KNN                                                      ", variable=algorithmVal, value=4).grid(column=1, row=7)
tkinter.Radiobutton(tab3, text="K-MEANS                                             ", variable=algorithmVal, value=5).grid(column=1, row=8)

ttk.Label(tab3, text='\t', font=('Helvetica', 14, 'bold')).grid(column=1, row=9)

tkinter.Button(tab3, text="Back", font=('calibre', 13, 'bold'), padx=25, pady=10, command=lambda: backTab(2)).grid(column=0, row=10, sticky='w')
tkinter.Button(tab3, text=" Go ", font=('calibre', 13, 'bold'), command=lambda: algorithmCall(), padx=25, pady=10).grid(column=2, row=10, sticky='e')


tab_control.pack(expand=1, fill='both')
window.resizable(width=False, height=False)
window.mainloop()



