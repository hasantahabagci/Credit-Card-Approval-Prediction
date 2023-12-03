import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plotdist(df, col, mx):
    """
    This function plots the distribution of a column
    :param df: dataframe
    :param col: column name
    :return: None
    """
    plt.figure(figsize=(10, 6))
    sns.distplot(df[col], kde=True)
    plt.vlines(df[col].mean(), 0, mx, colors='r', label='mean')
    plt.text(df[col].mean(), mx*1.01, "mean : "+ str(round(df[col].mean(),2)), fontsize=12)
    plt.show()

def plotbox(df, col):
    """
    This function plots the boxplot of a column
    :param df: dataframe
    :param col: column name
    :return: None
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(df[col])
    plt.show()