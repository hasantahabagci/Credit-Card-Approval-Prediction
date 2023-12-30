import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve

from sklearn.model_selection import GridSearchCV

def gridsearch(model, param_grid, X_train, y_train, X_test, y_test):
    """
    This function performs grid search
    :param model: model
    :param param_grid: parameter grid
    :param X_train: training data
    :param y_train: training labels
    :param X_test: test data
    :param y_test: test labels
    :return: best estimator
    """

    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_)
    print("Test set score: ", grid.score(X_test, y_test))

    return grid.best_estimator_

def confusionmatrix(y_test, y_pred):
    """
    This function plots the confusion matrix
    :param y_test: test labels
    :param y_pred: predicted labels
    :return: None
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()

    return cm

def plotroc(y_test, y_pred):
    """
    This function plots the ROC curve
    :param y_test: test labels
    :param y_pred: predicted labels
    :return: None
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plotpr(y_test, y_pred):
    """
    This function plots the precision-recall curve
    :param y_test: test labels
    :param y_pred: predicted labels
    :return: None
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()

def plotfeatureimportance(model, X_train):
    """
    This function plots the feature importance
    :param model: trained model
    :param X_train: training data
    :return: None
    """
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(X_train.columns, model.feature_importances_)
    plt.xticks(rotation=90)
    plt.show()

def plotlearningcurve(model, X_train, y_train):
    """
    This function plots the learning curve
    :param model: trained model
    :param X_train: training data
    :param y_train: training labels
    :return: None
    """
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, scoring='accuracy')

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label='train')
    plt.plot(train_sizes, test_scores.mean(axis=1), label='test')
    plt.legend()
    plt.show()

def printmetrics(y_test, y_pred, plot=False):
    """
    This function prints the metrics
    :param y_test: test labels
    :param y_pred: predicted labels
    :param plot: if True, plots the metrics
    :return: None
    """

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred))
    print("Precision: ", precision_score(y_test, y_pred))
    print("F1: ", f1_score(y_test, y_pred))
    print("ROC AUC: ", roc_auc_score(y_test, y_pred))
    print("MSE: ", mean_squared_error(y_test, y_pred))

    if plot:
        plt.figure(figsize=(10, 6))
        plt.bar(['Accuracy', 'Recall', 'Precision', 'F1', 'ROC AUC', 'MSE'], 
                [accuracy_score(y_test, y_pred), 
                 recall_score(y_test, y_pred), 
                 precision_score(y_test, y_pred), 
                 f1_score(y_test, y_pred), 
                 roc_auc_score(y_test, y_pred)])


