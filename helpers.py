import numpy as np
import pandas as pd

## Helper Functions for the dataset

#  Return the data without the header
def returnDataset(dataset: np.ndarray) -> np.ndarray:
    return dataset[1:]

#  Return the header of the dataset
def returnHeader(dataset: np.ndarray) -> np.ndarray:
    return dataset[0]

#  Return the class lables of the dataset
def returnClassLables(dataset: np.ndarray) -> np.ndarray:
    return dataset[:, -1]

#  Print the table using pandas for a cleaner looking table
def printTable(header: np.ndarray, dataset: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(dataset, columns=header)
    blankIndex=[''] * len(df)
    df.index=blankIndex
    return(df)