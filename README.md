# Decision-Tree-Classifier-AI

This project plays around with Decision Tree Classifiers with Entropy-Based Splitting. It uses the Scikit-Learn Python package. The user can input a dataset which will output a tree. The user can then input a sample for a prediction based on the tree.

To run the code, you can either run main.py, which contains hard coded data, or you can run ui.py which pops up a user interface with multiple input boxes.

You can use the following data to run the UI code:

Dataset:

[['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est', 'WillWait'],
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
['Yes', 'Yes', 'Yes', 'Yes', 'Full', '$', 'No', 'No', 'Burger', '30-60', 'Yes']]

Input:

['Yes', 'No', 'Yes', 'Yes', 'Some', '$', 'No', 'No', 'French', '30-60']
