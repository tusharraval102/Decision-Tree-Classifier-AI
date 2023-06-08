from collections import Counter
from sklearn import tree
from sklearn import preprocessing
from helpers import returnDataset, returnHeader, returnClassLables
import numpy as np
import pandas as pd
import graphviz


## Functions for the Decision Tree

#  Create a tree using an Entropy based decision tree classifier
def decisionTree(dataset: np.ndarray, labelEncoders: np.ndarray) -> tree.DecisionTreeClassifier:
    
    # Get the data and remove the feature names
    feature_names = returnHeader(dataset)
    data = returnDataset(dataset)

    # Transform the data into a format that can be used by the decision tree
    for i, le in enumerate(labelEncoders):
        data[:, i] = le.fit_transform(data[:, i])

    # Set the training input samples and the target values
    X = data[:, :-1]
    y = returnClassLables(data)

    print("**Transformed Dataset**\n")
    print(X)
    print("\n\n**Transformed Class Lables**\n")
    print(y)

    # Create the decision tree
    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(X, y)

    # Create a pretty tree using Graphviz to use in the application
    prettyTree(clf, feature_names, labelEncoders)

    return clf


# Create a pretty tree using Graphviz
def prettyTree(clf: tree.DecisionTreeClassifier, header: np.ndarray, labelEncoders: np.ndarray) -> None:
    length = header.__len__()-1
    newTree = tree.export_graphviz(clf, out_file=None, feature_names=header[0:length], class_names=labelEncoders[-1].classes_, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(newTree, format="png")
    graph.render("DecisionTree")

#  Classify a given set of instances
def classify(clf: tree.DecisionTreeClassifier, input: np.ndarray, labelEncoders: np.ndarray) -> str:
    prediction = clf.predict(input)
    print("Prediction: ", prediction)
    return str(labelEncoders[-1].classes_[prediction[0].astype(int)])
    
#  Convert the input to a format that can be used by the decision tree
def convertInput(input: np.ndarray, labelEncoders: np.ndarray) -> np.ndarray:
    new_data = np.array(input).reshape(1, -1)
    for i, le in enumerate(labelEncoders[:-1]):
        new_data[:, i] = le.transform(new_data[:, i])
    return new_data

#  Calculate the entropy of a given set of instances
def entropy(classLables: np.ndarray) -> float:
    totalInstances = classLables.__len__()
    classCounter = Counter(classLables)

    # Calculate the entropy
    entropy = 0
    for count in classCounter.values():
        probability = count/totalInstances
        entropy += (-probability * np.log2(probability))
    return entropy