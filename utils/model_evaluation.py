import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
)

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate a trained model and return various performance metrics
    
    Parameters:
    -----------
    model : object
        Trained model with predict and predict_proba methods
    X_train : pandas.DataFrame
        Training feature matrix
    X_test : pandas.DataFrame
        Testing feature matrix
    y_train : pandas.Series
        Training target vector
    y_test : pandas.Series
        Testing target vector
        
    Returns:
    --------
    dict
        Dictionary with various performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Return all metrics
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc,
        'precision_curve': precision,
        'recall_curve': recall,
        'avg_precision': avg_precision
    }

def confusion_matrix_plot(confusion_matrix):
    """
    Plot a confusion matrix using plotly
    
    Parameters:
    -----------
    confusion_matrix : numpy.ndarray
        Confusion matrix to plot
    """
    # Extract values
    tn, fp, fn, tp = confusion_matrix.ravel()
    
    # Labels
    categories = ['Non-Churn', 'Churn']
    
    # Normalize to get percentages
    total = tn + fp + fn + tp
    percent_tn = tn / total * 100
    percent_fp = fp / total * 100
    percent_fn = fn / total * 100
    percent_tp = tp / total * 100
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=[[percent_tn, percent_fp], [percent_fn, percent_tp]],
        x=['Predicted Non-Churn', 'Predicted Churn'],
        y=['Actual Non-Churn', 'Actual Churn'],
        text=[[f'TN: {tn} ({percent_tn:.1f}%)', f'FP: {fp} ({percent_fp:.1f}%)'], 
              [f'FN: {fn} ({percent_fn:.1f}%)', f'TP: {tp} ({percent_tp:.1f}%)']],
        texttemplate="%{text}",
        colorscale='Reds',
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=600,
        height=500
    )
    
    st.plotly_chart(fig)
    
    # Display metrics derivable from confusion matrix
    accuracy = (tn + tp) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("True Negatives:", tn)
        st.write("False Positives:", fp)
        st.write("False Negatives:", fn)
        st.write("True Positives:", tp)
    
    with col2:
        st.write("Accuracy:", f"{accuracy:.4f}")
        st.write("Precision:", f"{precision:.4f}")
        st.write("Recall (Sensitivity):", f"{recall:.4f}")
        st.write("Specificity:", f"{specificity:.4f}")

def roc_curve_plot(fpr, tpr, auc_score):
    """
    Plot a ROC curve using plotly
    
    Parameters:
    -----------
    fpr : array
        False positive rates
    tpr : array
        True positive rates
    auc_score : float
        Area under the ROC curve
    """
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.4f})',
        line=dict(color='blue', width=2)
    ))
    
    # Add diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=700,
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig)

def precision_recall_curve_plot(precision, recall, avg_precision):
    """
    Plot a precision-recall curve using plotly
    
    Parameters:
    -----------
    precision : array
        Precision values
    recall : array
        Recall values
    avg_precision : float
        Average precision score
    """
    fig = go.Figure()
    
    # Add precision-recall curve
    fig.add_trace(go.Scatter(
        x=recall, 
        y=precision,
        mode='lines',
        name=f'Precision-Recall Curve (AP = {avg_precision:.4f})',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=700,
        height=500,
        showlegend=True
    )
    
    # Add shaded area
    fig.add_trace(go.Scatter(
        x=np.concatenate([recall, [recall[-1], recall[0]]]),
        y=np.concatenate([precision, [0, precision[0]]]),
        fill='tozeroy',
        fillcolor='rgba(0,100,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False
    ))
    
    st.plotly_chart(fig)
