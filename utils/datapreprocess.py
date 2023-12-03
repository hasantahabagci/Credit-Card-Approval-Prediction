import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import LabelEncoder


def isCategorical(df, col, threshold):
    """
    This function decides if a column is categorical or not
    :param df: dataframe
    :param col: column name
    :param threshold: threshold to decide if a column is categorical or not
    :return: True if the column is categorical, False otherwise
    """

    if df[col].nunique() <= threshold:
        return True
    else:
        return False
    
def isNumerical(df, col):
    """
    This function decides if a column is numerical or not
    :param df: dataframe
    :param col: column name
    :param threshold: threshold to decide if a column is numerical or not
    :return: True if the column is numerical, False otherwise
    """
    numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    if df[col].dtype in numeric_types:
        return True
    else:
        return False

def labelEncoding(df, col):
    """
    This function encodes categorical columns
    :param df: dataframe
    :param col: column name
    :return: encoded column if inverse=False, decoded column if inverse=True
    """

    le = LabelEncoder()

    return le.fit_transform(df[col])
    

def oneHotEncoding(df, col):
    """
    This function encodes categorical columns
    :param df: dataframe
    :param col: column name
    :return: encoded column if inverse=False, decoded column if inverse=True
    """

    return pd.get_dummies(df[col], prefix=col)