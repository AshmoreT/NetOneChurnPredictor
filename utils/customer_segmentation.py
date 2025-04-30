import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def segment_customers(model, X, original_data):
    """
    Segment customers based on their churn probability
    
    Parameters:
    -----------
    model : object
        Trained model with predict_proba method
    X : pandas.DataFrame
        Feature matrix
    original_data : pandas.DataFrame
        Original dataset with all customer information
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with customer segments
    """
    # Predict churn probability
    churn_prob = model.predict_proba(X)[:, 1]
    
    # Add prediction to original data
    result_df = original_data.copy()
    result_df['Churn_Probability'] = churn_prob
    
    # Add CustomerID column if it doesn't exist
    if 'CustomerID' not in result_df.columns:
        result_df['CustomerID'] = [f"CUST_{i}" for i in range(len(result_df))]
    
    # Handle personal information columns - check if they exist and create placeholders if not
    # This ensures the system works offline without requiring real personal data
    personal_info_cols = ['First_Name', 'Surname', 'Street_Address', 'Gender', 'Location', 'Activation_Date', 'Plan_Type']
    for col in personal_info_cols:
        if col not in result_df.columns:
            # Create generic placeholder data that's clearly marked as non-personal data
            if col in ['First_Name', 'Surname', 'Street_Address', 'Gender', 'Location']:
                result_df[col] = "N/A"  # Non-identifiable placeholder for text fields
            elif col == 'Activation_Date':
                result_df[col] = "2023-01-01"  # Default date
            elif col == 'Plan_Type':
                result_df[col] = "Standard"  # Default plan
    
    # Segment customers based on churn probability
    result_df['Risk_Category'] = pd.cut(
        result_df['Churn_Probability'],
        bins=[0, 0.3, 0.7, 1],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    # Count by risk category
    risk_counts = result_df['Risk_Category'].value_counts().reset_index()
    risk_counts.columns = ['Risk_Category', 'Count']
    
    # Display risk distribution
    st.subheader("Customer Risk Distribution")
    
    fig = px.pie(
        risk_counts, 
        values='Count', 
        names='Risk_Category',
        title='Customer Distribution by Churn Risk',
        color='Risk_Category',
        color_discrete_map={
            'Low Risk': 'green',
            'Medium Risk': 'orange',
            'High Risk': 'red'
        }
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig)
    
    # Display risk statistics
    st.subheader("Risk Category Statistics")
    
    risk_stats = result_df.groupby('Risk_Category')['Churn_Probability'].describe().round(3)
    st.dataframe(risk_stats)
    
    return result_df

def plot_customer_segments(segments_df):
    """
    Create visualizations for customer segments
    
    Parameters:
    -----------
    segments_df : pandas.DataFrame
        DataFrame with customer segments
    """
    # Histogram of churn probabilities
    fig = px.histogram(
        segments_df,
        x='Churn_Probability',
        color='Risk_Category',
        title='Distribution of Churn Probabilities',
        nbins=50,
        color_discrete_map={
            'Low Risk': 'green',
            'Medium Risk': 'orange',
            'High Risk': 'red'
        }
    )
    
    fig.update_layout(
        xaxis_title='Churn Probability',
        yaxis_title='Count',
        legend_title='Risk Category',
        height=500,
        width=800
    )
    
    st.plotly_chart(fig)
    
    # Try to perform clustering if enough numerical features are available
    numerical_cols = segments_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != 'Churn_Probability']
    
    if len(numerical_cols) >= 2:
        # Select a subset of features for clustering
        top_features = numerical_cols[:min(10, len(numerical_cols))]
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(segments_df[top_features])
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled_data)
        
        # Create DataFrame for plotting
        pca_df = pd.DataFrame({
            'PC1': components[:, 0],
            'PC2': components[:, 1],
            'Risk_Category': segments_df['Risk_Category']
        })
        
        # Create scatter plot
        st.subheader("Customer Segmentation (PCA)")
        
        fig = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='Risk_Category',
            title='Customer Segments (PCA Projection)',
            color_discrete_map={
                'Low Risk': 'green',
                'Medium Risk': 'orange',
                'High Risk': 'red'
            }
        )
        
        fig.update_layout(
            height=600,
            width=800
        )
        
        st.plotly_chart(fig)
        
        # Try K-means clustering
        st.subheader("K-means Clustering")
        
        k = st.slider("Number of clusters", 2, 8, 3)
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add clusters to PCA data
        pca_df['Cluster'] = clusters
        
        # Create scatter plot with clusters
        fig = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='Cluster',
            title=f'K-means Clustering (k={k})',
            hover_data=['Risk_Category'],
            color_continuous_scale=px.colors.qualitative.G10
        )
        
        fig.update_layout(
            height=600,
            width=800
        )
        
        st.plotly_chart(fig)
        
        # Analyze clusters
        segments_df['Cluster'] = clusters
        
        cluster_stats = segments_df.groupby('Cluster')['Churn_Probability'].mean().reset_index()
        cluster_stats.columns = ['Cluster', 'Average Churn Probability']
        
        fig = px.bar(
            cluster_stats,
            x='Cluster',
            y='Average Churn Probability',
            title='Average Churn Probability by Cluster',
            color='Average Churn Probability',
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            xaxis_title='Cluster',
            yaxis_title='Average Churn Probability',
            height=500,
            width=700
        )
        
        st.plotly_chart(fig)
        
        # Show cluster characteristics - safely get only available columns
        st.subheader("Cluster Characteristics")
        available_features = [feat for feat in top_features if feat in segments_df.columns]
        if available_features:
            cluster_profiles = segments_df.groupby('Cluster')[available_features].mean().reset_index()
            st.dataframe(cluster_profiles)
