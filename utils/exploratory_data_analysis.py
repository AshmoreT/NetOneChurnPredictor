import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_churn_distribution(data, target_column):
    """
    Plot the distribution of the churn target variable
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataset
    target_column : str
        The name of the target column
    """
    # Count the target values
    target_counts = data[target_column].value_counts().reset_index()
    target_counts.columns = ['Churn', 'Count']
    
    # Convert to string for better display
    target_counts['Churn'] = target_counts['Churn'].astype(str)
    
    # Calculate percentages
    total = target_counts['Count'].sum()
    target_counts['Percentage'] = (target_counts['Count'] / total * 100).round(2)
    
    # Create a pie chart
    fig1 = px.pie(
        target_counts, 
        values='Count', 
        names='Churn',
        title='Customer Churn Distribution',
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.4
    )
    
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig1)
    
    # Create a bar chart
    fig2 = px.bar(
        target_counts,
        x='Churn',
        y='Count',
        text='Count',
        color='Churn',
        title='Customer Churn Count',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig2.update_layout(
        xaxis_title='Churn',
        yaxis_title='Count',
        showlegend=False
    )
    
    fig2.update_traces(texttemplate='%{text}', textposition='outside')
    st.plotly_chart(fig2)
    
    # Display a table with counts and percentages
    st.write("Churn Distribution")
    st.dataframe(target_counts)

def plot_feature_correlation(data):
    """
    Plot correlation matrix for numerical features
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataset
    """
    # Get numerical columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(numerical_cols) > 1:
        # Calculate correlation matrix
        corr_matrix = data[numerical_cols].corr()
        
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
        
        # Plot interactive correlation heatmap using plotly
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Interactive Correlation Matrix",
            labels=dict(color="Correlation")
        )
        
        fig.update_layout(width=800, height=800)
        st.plotly_chart(fig)
        
        # Show highest correlations
        st.subheader("Highest Correlations")
        
        # Get the correlations in a table format
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        # Convert to dataframe and sort
        corr_df = pd.DataFrame(corr_pairs)
        corr_df['Absolute Correlation'] = corr_df['Correlation'].abs()
        corr_df = corr_df.sort_values('Absolute Correlation', ascending=False).head(10)
        
        st.dataframe(corr_df[['Feature 1', 'Feature 2', 'Correlation']])
    else:
        st.info("Not enough numerical features to calculate correlations.")

def plot_categorical_features(data, categorical_columns, target_column):
    """
    Plot the relationship between categorical features and the target
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataset
    categorical_columns : list
        List of categorical columns to analyze
    target_column : str
        The name of the target column
    """
    for col in categorical_columns:
        # Create a count plot
        st.subheader(f"{col} vs {target_column}")
        
        # Calculate counts by category and target
        counts = data.groupby([col, target_column]).size().reset_index(name='Count')
        
        # Create a grouped bar chart
        fig = px.bar(
            counts, 
            x=col, 
            y='Count', 
            color=target_column,
            barmode='group',
            title=f"{col} Distribution by {target_column}",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            xaxis_title=col,
            yaxis_title='Count'
        )
        
        st.plotly_chart(fig)
        
        # Calculate churn rate by category
        category_counts = data[col].value_counts()
        churn_rate = data.groupby(col)[target_column].mean().reset_index()
        churn_rate.columns = [col, 'Churn Rate']
        
        # Sort by churn rate
        churn_rate = churn_rate.sort_values('Churn Rate', ascending=False)
        
        # Create a bar chart for churn rates
        fig2 = px.bar(
            churn_rate,
            x=col,
            y='Churn Rate',
            color='Churn Rate',
            title=f"Churn Rate by {col}",
            color_continuous_scale=px.colors.sequential.Reds
        )
        
        fig2.update_layout(
            xaxis_title=col,
            yaxis_title='Churn Rate',
            showlegend=False
        )
        
        st.plotly_chart(fig2)
        
        # Add a separator
        st.markdown("---")

def plot_numerical_features(data, numerical_columns, target_column):
    """
    Plot the relationship between numerical features and the target
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataset
    numerical_columns : list
        List of numerical columns to analyze
    target_column : str
        The name of the target column
    """
    for col in numerical_columns:
        st.subheader(f"{col} vs {target_column}")
        
        # Create distribution plot
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=["Distribution", "Box Plot"])
        
        # Distribution plot
        for target_val in data[target_column].unique():
            subset = data[data[target_column] == target_val]
            fig.add_trace(
                go.Histogram(
                    x=subset[col],
                    name=f"{target_column}={target_val}",
                    opacity=0.7,
                    nbinsx=30
                ),
                row=1, col=1
            )
        
        # Box plot
        for target_val in data[target_column].unique():
            subset = data[data[target_column] == target_val]
            fig.add_trace(
                go.Box(
                    y=subset[col],
                    name=f"{target_column}={target_val}"
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title=f"{col} Analysis by {target_column}",
            barmode='overlay',
            height=500,
            width=900
        )
        
        st.plotly_chart(fig)
        
        # Statistical summary
        st.write("Statistical Summary by Target")
        summary = data.groupby(target_column)[col].describe().reset_index()
        st.dataframe(summary)
        
        # Add a separator
        st.markdown("---")
