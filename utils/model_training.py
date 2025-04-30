import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import streamlit as st

def train_logistic_regression(X, y, test_size=0.2, random_state=42, C=1.0, solver='liblinear', max_iter=100):
    """
    Train a logistic regression model
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    test_size : float
        Proportion of the dataset to include in the test split
    random_state : int
        Random seed for reproducibility
    C : float
        Inverse of regularization strength
    solver : str
        Algorithm to use in the optimization problem
    max_iter : int
        Maximum number of iterations
        
    Returns:
    --------
    tuple
        (model, X_train, X_test, y_train, y_test)
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Initialize and train the model
    model = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def train_random_forest(X, y, test_size=0.2, random_state=42, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """
    Train a random forest model
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    test_size : float
        Proportion of the dataset to include in the test split
    random_state : int
        Random seed for reproducibility
    n_estimators : int
        Number of trees in the forest
    max_depth : int or None
        Maximum depth of the trees
    min_samples_split : int
        Minimum number of samples required to split an internal node
    min_samples_leaf : int
        Minimum number of samples required to be at a leaf node
        
    Returns:
    --------
    tuple
        (model, X_train, X_test, y_train, y_test)
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Initialize and train the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def train_xgboost(X, y, test_size=0.2, random_state=42, n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8):
    """
    Train an XGBoost model
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    test_size : float
        Proportion of the dataset to include in the test split
    random_state : int
        Random seed for reproducibility
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum depth of the trees
    learning_rate : float
        Boosting learning rate
    subsample : float
        Subsample ratio of the training instances
        
    Returns:
    --------
    tuple
        (model, X_train, X_test, y_train, y_test)
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Initialize and train the model
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test
