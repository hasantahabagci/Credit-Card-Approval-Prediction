import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.model_selection import learning_curve, GridSearchCV, validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier



from imblearn.over_sampling import SMOTE

def gridSearch(model, param_grid, X_train, y_train, cv=5, scoring='accuracy'):
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

    grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
    grid.fit(X_train, y_train)


    return grid

def randomForest(X_train, y_train, n_estimators=100, max_depth=10, max_features='auto', min_samples_split=2, min_samples_leaf=1, bootstrap=True):
    """
    This function performs random forest
    :param X_train: training data
    :param y_train: training labels
    :param X_test: test data
    :param y_test: test labels
    :param n_estimators: number of trees
    :param max_depth: max depth of the tree
    :param max_features: max number of features
    :param min_samples_split: min number of samples to split
    :param min_samples_leaf: min number of samples in a leaf
    :param bootstrap: if True, bootstrap samples are used
    :return: predicted labels
    """

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, bootstrap=bootstrap)
    rf.fit(X_train, y_train)


    return rf

def smote(X_train, y_train):
    """
    This function performs SMOTE
    :param X_train: training data
    :param y_train: training labels
    :return: oversampled training data and labels
    """

    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    print("Before SMOTE shape: ", X_train.shape, y_train.shape)
    print("After SMOTE shape: ", X_train_sm.shape, y_train_sm.shape)

    return X_train_sm, y_train_sm


def decisionTree(X_train, y_train, max_depth=10, max_features='auto', min_samples_split=2, min_samples_leaf=1):
    """
    This function performs decision tree
    :param X_train: training data
    :param y_train: training labels
    :param X_test: test data
    :param y_test: test labels
    :param max_depth: max depth of the tree
    :param max_features: max number of features
    :param min_samples_split: min number of samples to split
    :param min_samples_leaf: min number of samples in a leaf
    :return: predicted labels
    """

    dt = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    dt.fit(X_train, y_train)


    return dt


def logisticRegression(X_train, y_train, C=1.0, penalty='l2', solver='lbfgs'):
    """
    This function performs logistic regression
    :param X_train: training data
    :param y_train: training labels
    :param X_test: test data
    :param y_test: test labels
    :param C: penalty parameter
    :param penalty: penalty type
    :param solver: solver type
    :return: predicted labels
    """

    lr = LogisticRegression(C=C, penalty=penalty, solver=solver)
    lr.fit(X_train, y_train)


    return lr

def xgboost(X_train, y_train, max_depth=10, learning_rate=0.1, n_estimators=100, objective='binary:logistic', booster='gbtree', n_jobs=1):
    """
    This function performs XGBoost
    :param X_train: training data
    :param y_train: training labels
    :param X_test: test data
    :param y_test: test labels
    :param max_depth: max depth of the tree
    :param learning_rate: learning rate
    :param n_estimators: number of trees
    :param objective: objective function
    :param booster: booster type
    :param n_jobs: number of jobs
    :return: predicted labels
    """

    xgb = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, objective=objective, booster=booster, n_jobs=n_jobs)
    xgb.fit(X_train, y_train)


    return xgb


def adaBoost(X_train, y_train, n_estimators=100, learning_rate=1.0, algorithm='SAMME.R'):
    """
    This function performs AdaBoost
    :param X_train: training data
    :param y_train: training labels
    :param X_test: test data
    :param y_test: test labels
    :param n_estimators: number of trees
    :param learning_rate: learning rate
    :param algorithm: algorithm type
    :return: predicted labels
    """

    ada = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm)
    ada.fit(X_train, y_train)


    return ada


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


def plotroc(y_test, y_pred, figsize=(5, 3)):
    """
    This function plots the ROC curve
    :param y_test: test labels
    :param y_pred: predicted labels
    :return: None
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plotpr(y_test, y_pred, figsize=(5, 3)):
    """
    This function plots the precision-recall curve
    :param y_test: test labels
    :param y_pred: predicted labels
    :return: None
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=figsize)
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

def plotlearningcurve(model, X_train, y_train, figsize=(5, 3)):
    """
    This function plots the learning curve
    :param model: trained model
    :param X_train: training data
    :param y_train: training labels
    :return: None
    """
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, scoring='accuracy')

    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_scores.mean(axis=1), label='train')
    plt.plot(train_sizes, test_scores.mean(axis=1), label='test')
    plt.legend()
    plt.show()

def plotValidationCurve(model, param_name, param_range, X_train, y_train, figsize=(5, 3)):
    """
    This function plots the validation curve
    :param model: trained model
    :param param_name: parameter name
    :param param_range: parameter range
    :param X_train: training data
    :param y_train: training labels
    :return: None
    """
    train_scores, test_scores = validation_curve(model, X_train, y_train, param_name=param_name, param_range=param_range, cv=5, scoring='accuracy')

    plt.figure(figsize=figsize)
    plt.plot(param_range, train_scores.mean(axis=1), label='train')
    plt.plot(param_range, test_scores.mean(axis=1), label='test')
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


