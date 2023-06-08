# Import all relevent modules for Decision Tree Classification
from sklearn import preprocessing, tree
from helpers import returnDataset, returnHeader, returnClassLables, printTable
from decisionTree import decisionTree, classify, convertInput
import numpy as np

# Main function of the program
def main():
    # Hardcoded dataset for testing
    dataset = np.array([
        ['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est', 'WillWait'],
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
    
    # Hardcoded unseen instance for testing
    unseenInstance = np.array(['Yes', 'No', 'Yes', 'Yes', 'Some', '$', 'No', 'No', 'French', '30-60'])

    labelEncoders = [preprocessing.LabelEncoder() for i in range(len(dataset[0]))]
    
    print("**Dataset**\n")
    newDataset = returnDataset(dataset)
    print(newDataset)

    print("\n\n**Header**\n")
    header = returnHeader(dataset)
    print(header)
    
    print("\n\n**Pretty Table**\n")
    prettyTable = printTable(dataset[0], newDataset)
    print(prettyTable)
    
    print("\n\n**Class Lables**\n")
    classLables = returnClassLables(newDataset)
    print(classLables)
    
    print("\n\n**Decision Tree**\n")
    decision = decisionTree(dataset, labelEncoders)
    
    print("\n\n**Unseen Instance Transformed**\n")
    unseenInstanceConverted = convertInput(unseenInstance, labelEncoders)
    print(str(unseenInstance) + " -> " + str(unseenInstanceConverted))

    print("\n\n**Classified Unseen Instance**\n")
    newArray = np.array(unseenInstanceConverted)
    classifiedInstance = classify(decision, unseenInstanceConverted, labelEncoders)
    print("Prediction: " + classifiedInstance)
    
    


# Run the main function of the program
main()