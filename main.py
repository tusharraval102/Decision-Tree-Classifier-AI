# Import all relevent modules for Decision Tree Classification
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
import graphviz

#  Create a tree using an Entropy based decision tree classifier

def decisionTree(header: np.ndarray, dataset: np.ndarray) -> None:
    # Find the last column of the dataset - which should be the class column
    lastColumn = dataset.__len__()-2

    # Split the dataset into X and Y
    X = dataset[:, 0:lastColumn]
    Y = dataset[:, lastColumn]

    # Encode the data
    le = preprocessing.LabelEncoder()
    
    # Check if the data is a string or not and transform it
    for i in range(0, lastColumn):
        if(type(X[:, i]) == 'int' or type(X[:, i]) == 'float'):
            continue
        else:
            X[:, i] = le.fit_transform(X[:, i])
    
    # Transform the Y data
    Y = le.fit_transform(Y)

    # Create the decision tree
    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(X, Y)

    # Create the tree using graphviz for export
    newTree = tree.export_graphviz(clf, out_file=None, feature_names=header[1:11], class_names=le.classes_, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(newTree)
    graph.render("Decision Tree")


#  Check if the input is valid
def inputIsValid(dataset):
    return True

#  Calculate the entropy of a given set of instances
def entropy():
    return "null"

#  Calculate the information gain of a given set of instances
def split():
    return "null"


#  Classify a given set of instances
def classify():
    return "null"


#  Create a user interface for the program
def userInterface():
    return "null"

def returnData(dataset: np.ndarray) -> np.ndarray:
    newDataset = np.delete(dataset, 0,0)
    return newDataset

def printTable(header: np.ndarray, dataset: np.ndarray) -> None:
    df = pd.DataFrame(dataset, columns=header)
    blankIndex=[''] * len(df)
    df.index=blankIndex
    print(df)

def main():
    dataset = np.array([
        ['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res Type', 'Est', 'Wait Time', 'WillWait'],
        ['Yes', 'No', 'No', 'Yes', 'Some', '$$$', 'No', 'Yes', 'French', '0-10', 'Yes'],
        ['Yes', 'No', 'No', 'Yes', 'Full', '$', 'No', 'No', 'Thai', '30-60', 'No'],
        ['No', 'Yes', 'No', 'No', 'Some', '$', 'No', 'No', 'Burger', '0-10', 'Yes'],
        ['Yes', 'No', 'Yes', 'Yes', 'Full', '$', 'No', 'No', 'Thai', '10-30', 'Yes'],
        ['Yes', 'No', 'Yes', 'No', 'Full', '$$$', 'No', 'Yes', 'French', '>60', 'No'],
        ['No', 'Yes', 'No', 'Yes', 'Some', '$$', 'Yes', 'Yes', 'Italian', '0-10', 'Yes'],
        ['No', 'Yes', 'No', 'No', 'None', '$', 'Yes', 'No', 'Burger', '0-10', 'No'],
        ['No', 'No', 'No', 'Yes', 'Some', '$$', 'Yes', 'Yes', 'Thai', '0-10', 'Yes'],
        ['No', 'Yes', 'Yes', 'No', 'Full', '$', 'Yes', 'No', 'Burger', '>60', 'No'],
        ['Yes', 'Yes', 'Yes', 'Yes', 'Full', '$$$', 'No', 'Yes', 'Italian', '10-30', 'No'],
        ['No', 'No', 'No', 'No', 'None', '$', 'No', 'No', 'Thai', '0-10', 'No'],
        ['Yes', 'Yes', 'Yes', 'Yes', 'Full', '$', 'No', 'No', 'Burger', '30-60', 'Yes']
        ])
    
    if(inputIsValid(dataset)):
        newDataset = returnData(dataset)
        # printTable(dataset[0], newDataset)
        decisionTree(dataset[0],newDataset)
    
    #  Create a numpy array from the dataset
    # X = dataset[:, 1:11]  # 
    # y = dataset[:, 11]



main()


