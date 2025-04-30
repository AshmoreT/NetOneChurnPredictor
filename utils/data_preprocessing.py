import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import streamlit as st

def handle_missing_values(data, numerical_strategy='mean', categorical_strategy='mode'):
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataset
    numerical_strategy : str
        Strategy to handle missing values in numerical columns ('mean', 'median', 'mode', 'drop')
    categorical_strategy : str
        Strategy to handle missing values in categorical columns ('mode', 'drop', 'new_category')
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with handled missing values
    """
    # Make a copy of the data
    df = data.copy()
    
    # Get numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Handle missing values in numerical columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            if numerical_strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif numerical_strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif numerical_strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif numerical_strategy == 'drop':
                df.dropna(subset=[col], inplace=True)
    
    # Handle missing values in categorical columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            if categorical_strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif categorical_strategy == 'drop':
                df.dropna(subset=[col], inplace=True)
            elif categorical_strategy == 'new_category':
                df[col].fillna('Unknown', inplace=True)
    
    return df

def encode_categorical_features(data, encoding_strategy='one-hot', categorical_columns=None):
    """
    Encode categorical features in the dataset
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataset
    encoding_strategy : str
        Strategy to encode categorical features ('one-hot', 'label')
    categorical_columns : list
        List of categorical columns to encode. If None, all object and category columns will be encoded.
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with encoded categorical features
    """
    # Make a copy of the data
    df = data.copy()
    
    # If categorical_columns is not provided, get all categorical columns
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        # Filter only existing categorical columns
        categorical_columns = [col for col in categorical_columns if col in df.columns and df[col].dtype in ['object', 'category']]
    
    # Encode categorical features
    if encoding_strategy == 'one-hot':
        # Use pandas get_dummies for one-hot encoding
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    elif encoding_strategy == 'label':
        # Use LabelEncoder for label encoding
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    return df

def preprocess_data(data, target_column, feature_columns=None):
    """
    Preprocess the data for model training
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataset
    target_column : str
        The name of the target column
    feature_columns : list
        List of feature columns to use. If None, all columns except target will be used.
        
    Returns:
    --------
    tuple
        (X, y, preprocessed_data) where X is the feature matrix, y is the target vector,
        and preprocessed_data is the full preprocessed dataset
    """
    # Make a copy of the data
    df = data.copy()
    
    # If feature_columns is not provided, use all columns except target
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    # Separate features and target
    X = df[feature_columns]
    y = df[target_column]
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if numerical_cols:
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y, df
