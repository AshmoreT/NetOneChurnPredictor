import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

def feature_importance_plot(feature_importance_df):
    """
    Plot feature importance
    
    Parameters:
    -----------
    feature_importance_df : pandas.DataFrame
        DataFrame with Feature and Importance columns
    """
    # Sort by importance for better visualization
    sorted_df = feature_importance_df.sort_values('Importance', ascending=True)
    
    # Create a horizontal bar chart
    fig = px.bar(
        sorted_df.tail(15),  # Take top 15 features
        y='Feature',
        x='Importance',
        orientation='h',
        title='Feature Importance',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=600,
        width=800
    )
    
    st.plotly_chart(fig)
    
    # Display the importance values
    st.subheader("Feature Importance Values")
    st.dataframe(sorted_df.sort_values('Importance', ascending=False))

def shap_analysis(model, X, model_name):
    """
    Perform SHAP analysis for model interpretation
    
    Parameters:
    -----------
    model : object
        Trained model
    X : pandas.DataFrame
        Feature matrix
    model_name : str
        Name of the model for display purposes
    """
    # Sample data for SHAP analysis if dataset is too large
    if X.shape[0] > 1000:
        X_sample = X.sample(n=1000, random_state=42)
    else:
        X_sample = X
    
    # Create SHAP explainer based on model type
    if model_name == "Logistic Regression":
        explainer = shap.LinearExplainer(model, X_sample)
    elif model_name in ["Random Forest", "XGBoost"]:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X_sample)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # If shap_values is a list (for multi-class), take the values for class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Convert to DataFrame for easier manipulation
    shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)
    
    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_df).mean().sort_values(ascending=False)
    
    # Create bar plot for mean absolute SHAP values
    st.subheader("SHAP Feature Importance")
    
    fig = px.bar(
        x=mean_abs_shap.values,
        y=mean_abs_shap.index,
        orientation='h',
        title='Mean |SHAP Value| (impact on model output)',
        labels={'x': 'mean |SHAP value|', 'y': 'Feature'},
        color=mean_abs_shap.values,
        color_continuous_scale='RdBu_r'
    )
    
    fig.update_layout(height=600, width=800)
    st.plotly_chart(fig)
    
    # Create summary plot using matplotlib
    st.subheader("SHAP Summary Plot")
    
    # Save the figure to display with Streamlit
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    st.pyplot(fig)
    
    # SHAP dependence plots for top features
    st.subheader("SHAP Dependence Plots")
    st.write("These plots show how the feature value affects the model prediction.")
    
    top_features = mean_abs_shap.head(5).index.tolist()
    
    for feature in top_features:
        st.write(f"### Dependence Plot for {feature}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, X_sample, show=False)
        st.pyplot(fig)
    
    # SHAP individual prediction explanation
    st.subheader("Individual Prediction Explanation")
    
    # Let user select a sample to explain
    sample_index = st.slider("Select a sample to explain", 0, len(X_sample) - 1, 0)
    
    # Display the selected sample
    st.write("Selected Sample Features:")
    st.dataframe(X_sample.iloc[sample_index:sample_index+1])
    
    # Plot the SHAP values for the selected sample
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.force_plot(
        explainer.expected_value,
        shap_values[sample_index, :],
        X_sample.iloc[sample_index, :],
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)
