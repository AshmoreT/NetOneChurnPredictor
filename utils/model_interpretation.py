import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

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
    Alternative feature importance analysis without SHAP library
    
    Parameters:
    -----------
    model : object
        Trained model
    X : pandas.DataFrame
        Feature matrix
    model_name : str
        Name of the model for display purposes
    """
    st.subheader("Feature Importance Analysis")
    
    # Sample data if dataset is too large
    if X.shape[0] > 1000:
        X_sample = X.sample(n=1000, random_state=42)
    else:
        X_sample = X
    
    # Get feature importance based on model type
    feature_importance = None
    
    if model_name == "Logistic Regression":
        # For logistic regression, we use the absolute coefficients
        if hasattr(model, 'coef_'):
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': np.abs(model.coef_[0])
            })
    
    elif model_name in ["Random Forest", "XGBoost"]:
        # For tree-based models, use the feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            })
    
    if feature_importance is not None:
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Plot feature importance
        fig = px.bar(
            feature_importance.head(15),
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Feature Importance for {model_name}',
            color='Importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=600,
            width=800
        )
        
        st.plotly_chart(fig)
        
        # Display importance values
        st.subheader("Feature Importance Values")
        st.dataframe(feature_importance)
        
        # Feature correlation with importance
        st.subheader("Feature Correlation Analysis")
        
        # Calculate correlation matrix
        corr_matrix = X_sample.corr()
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        heatmap = sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            ax=ax
        )
        plt.title("Feature Correlation Matrix")
        st.pyplot(fig)
        
        # Top correlated features
        top_features = feature_importance['Feature'].head(5).tolist()
        
        # Plot distributions of top features
        st.subheader("Top Feature Distributions")
        
        for feature in top_features:
            if feature in X_sample.columns:
                st.write(f"### Distribution of {feature}")
                
                fig = px.histogram(
                    X_sample,
                    x=feature,
                    title=f'Distribution of {feature}',
                    marginal='box',
                    nbins=30
                )
                
                st.plotly_chart(fig)
    else:
        st.warning("Feature importance not available for this model type.")
