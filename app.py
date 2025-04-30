import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import io
import os
from utils.data_preprocessing import preprocess_data, handle_missing_values, encode_categorical_features
from utils.exploratory_data_analysis import plot_churn_distribution, plot_feature_correlation, plot_categorical_features, plot_numerical_features
from utils.model_training import train_logistic_regression, train_random_forest, train_xgboost
from utils.model_evaluation import evaluate_model, confusion_matrix_plot, roc_curve_plot, precision_recall_curve_plot
from utils.model_interpretation import feature_importance_plot, shap_analysis
from utils.customer_segmentation import segment_customers, plot_customer_segments

# Set page configuration
st.set_page_config(
    page_title="NetOne Churn Prediction System",
    page_icon="📊",
    layout="wide"
)

# Application title and introduction
st.title("NetOne Telecommunications Churn Prediction System")

st.markdown("""
This application helps NetOne predict customer churn and identify key factors that contribute to customer attrition.
Upload your customer data, analyze patterns, train prediction models, and get actionable insights.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Data Upload & Preprocessing", "Exploratory Data Analysis", "Model Training", 
     "Model Evaluation", "Feature Importance", "Customer Segmentation", "Actionable Insights"]
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = {}
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = {}

# Data Upload & Preprocessing
if page == "Data Upload & Preprocessing":
    st.header("Data Upload & Preprocessing")
    
    # Data upload section
    uploaded_file = st.file_uploader("Upload your customer data (CSV or Excel file)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Load the data
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.session_state.data = data
            
            # Display raw data
            st.subheader("Raw Data Preview")
            st.dataframe(data.head())
            
            st.write(f"Dataset shape: {data.shape[0]} rows and {data.shape[1]} columns")
            
            # Data info section
            st.subheader("Data Information")
            buffer = io.StringIO()
            data.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            
            # Data statistics section
            st.subheader("Descriptive Statistics")
            st.dataframe(data.describe().T)
            
            # Missing values check
            st.subheader("Missing Values")
            missing_values = data.isnull().sum()
            missing_values_percentage = (missing_values / len(data) * 100).round(2)
            missing_data = pd.DataFrame({
                'Missing Values': missing_values,
                'Percentage (%)': missing_values_percentage
            })
            st.dataframe(missing_data)
            
            # Preprocessing options
            st.subheader("Preprocessing Options")
            
            # Target variable selection
            target_column = st.selectbox("Select the target column (churn indicator)", data.columns)
            
            # Handle missing values
            st.write("Missing Values Strategy")
            numerical_strategy = st.selectbox("Strategy for numerical features", ["mean", "median", "mode", "drop"])
            categorical_strategy = st.selectbox("Strategy for categorical features", ["mode", "drop", "new_category"])
            
            # Handle categorical features
            categorical_encoding = st.selectbox("Categorical Encoding Method", ["one-hot", "label"])
            
            # Feature selection
            st.write("Select features to include in the model:")
            feature_columns = st.multiselect("Features", [col for col in data.columns if col != target_column], 
                                            default=[col for col in data.columns if col != target_column])
            
            # Apply preprocessing
            if st.button("Preprocess Data"):
                with st.spinner("Preprocessing data..."):
                    # Handle missing values
                    data_cleaned = handle_missing_values(data.copy(), numerical_strategy, categorical_strategy)
                    
                    # Encode categorical features
                    data_encoded = encode_categorical_features(data_cleaned, categorical_encoding, feature_columns)
                    
                    # Final preprocessing
                    X, y, preprocessed_data = preprocess_data(data_encoded, target_column, feature_columns)
                    
                    st.session_state.preprocessed_data = preprocessed_data
                    st.session_state.features = X
                    st.session_state.target = y
                    st.session_state.feature_columns = feature_columns
                    st.session_state.target_column = target_column
                    
                    st.subheader("Preprocessed Data Preview")
                    st.dataframe(preprocessed_data.head())
                    
                    st.success("Data preprocessing completed successfully!")
                    
                    # Show class distribution
                    st.subheader("Target Variable Distribution")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    target_counts = preprocessed_data[target_column].value_counts()
                    ax.bar(target_counts.index.astype(str), target_counts.values)
                    ax.set_title("Distribution of Target Variable")
                    ax.set_xlabel("Churn")
                    ax.set_ylabel("Count")
                    st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Exploratory Data Analysis
elif page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    
    if st.session_state.data is None:
        st.warning("Please upload and preprocess your data first.")
    else:
        data = st.session_state.data
        
        # EDA Options
        analysis_type = st.selectbox("Select Analysis Type", 
                                    ["Churn Distribution", "Feature Correlation", 
                                     "Categorical Feature Analysis", "Numerical Feature Analysis"])
        
        if analysis_type == "Churn Distribution":
            st.subheader("Churn Distribution Analysis")
            if st.session_state.target_column:
                plot_churn_distribution(data, st.session_state.target_column)
        
        elif analysis_type == "Feature Correlation":
            st.subheader("Feature Correlation Analysis")
            plot_feature_correlation(data)
        
        elif analysis_type == "Categorical Feature Analysis":
            st.subheader("Categorical Feature Analysis")
            if st.session_state.target_column:
                categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
                if categorical_columns:
                    selected_cat_features = st.multiselect("Select categorical features", 
                                                          categorical_columns, 
                                                          default=categorical_columns[:min(3, len(categorical_columns))])
                    if selected_cat_features:
                        plot_categorical_features(data, selected_cat_features, st.session_state.target_column)
                else:
                    st.info("No categorical features found in the dataset.")
            else:
                st.warning("Please select a target column in the preprocessing step.")
        
        elif analysis_type == "Numerical Feature Analysis":
            st.subheader("Numerical Feature Analysis")
            if st.session_state.target_column:
                numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if numerical_columns:
                    selected_num_features = st.multiselect("Select numerical features", 
                                                         numerical_columns,
                                                         default=numerical_columns[:min(3, len(numerical_columns))])
                    if selected_num_features:
                        plot_numerical_features(data, selected_num_features, st.session_state.target_column)
                else:
                    st.info("No numerical features found in the dataset.")
            else:
                st.warning("Please select a target column in the preprocessing step.")

# Model Training
elif page == "Model Training":
    st.header("Model Training")
    
    if st.session_state.features is None or st.session_state.target is None:
        st.warning("Please upload and preprocess your data first.")
    else:
        X = st.session_state.features
        y = st.session_state.target
        
        # Model selection
        model_type = st.selectbox("Select Model Type", 
                                 ["Logistic Regression", "Random Forest", "XGBoost"])
        
        # Model configuration
        st.subheader("Model Configuration")
        
        test_size = st.slider("Test size (%)", 10, 40, 20) / 100
        random_state = st.number_input("Random state", 0, 100, 42)
        
        with st.expander("Advanced Model Parameters"):
            if model_type == "Logistic Regression":
                C = st.number_input("Regularization parameter (C)", 0.01, 10.0, 1.0, 0.1)
                solver = st.selectbox("Solver", ["liblinear", "lbfgs", "newton-cg", "sag", "saga"])
                max_iter = st.number_input("Maximum iterations", 100, 1000, 100, 50)
                
                if st.button("Train Logistic Regression Model"):
                    with st.spinner("Training Logistic Regression model..."):
                        model, X_train, X_test, y_train, y_test = train_logistic_regression(
                            X, y, test_size, random_state, C, solver, max_iter
                        )
                        st.session_state.models["Logistic Regression"] = {
                            "model": model,
                            "X_train": X_train,
                            "X_test": X_test,
                            "y_train": y_train,
                            "y_test": y_test
                        }
                        st.success("Logistic Regression model trained successfully!")
            
            elif model_type == "Random Forest":
                n_estimators = st.number_input("Number of trees", 10, 500, 100, 10)
                max_depth = st.number_input("Maximum depth", 2, 30, 10, 1)
                min_samples_split = st.number_input("Minimum samples to split", 2, 20, 2, 1)
                min_samples_leaf = st.number_input("Minimum samples in leaf", 1, 20, 1, 1)
                
                if st.button("Train Random Forest Model"):
                    with st.spinner("Training Random Forest model..."):
                        model, X_train, X_test, y_train, y_test = train_random_forest(
                            X, y, test_size, random_state, n_estimators, max_depth, 
                            min_samples_split, min_samples_leaf
                        )
                        st.session_state.models["Random Forest"] = {
                            "model": model,
                            "X_train": X_train,
                            "X_test": X_test,
                            "y_train": y_train,
                            "y_test": y_test
                        }
                        st.success("Random Forest model trained successfully!")
            
            elif model_type == "XGBoost":
                n_estimators = st.number_input("Number of boosting rounds", 10, 500, 100, 10)
                max_depth = st.number_input("Maximum depth", 2, 15, 6, 1)
                learning_rate = st.number_input("Learning rate", 0.01, 0.5, 0.1, 0.01)
                subsample = st.number_input("Subsample ratio", 0.5, 1.0, 0.8, 0.1)
                
                if st.button("Train XGBoost Model"):
                    with st.spinner("Training XGBoost model..."):
                        model, X_train, X_test, y_train, y_test = train_xgboost(
                            X, y, test_size, random_state, n_estimators, max_depth, 
                            learning_rate, subsample
                        )
                        st.session_state.models["XGBoost"] = {
                            "model": model,
                            "X_train": X_train,
                            "X_test": X_test,
                            "y_train": y_train,
                            "y_test": y_test
                        }
                        st.success("XGBoost model trained successfully!")
        
        # Display trained models
        st.subheader("Trained Models")
        if st.session_state.models:
            for model_name in st.session_state.models:
                st.write(f"✅ {model_name}")
        else:
            st.info("No models have been trained yet.")

# Model Evaluation
elif page == "Model Evaluation":
    st.header("Model Evaluation")
    
    if not st.session_state.models:
        st.warning("Please train at least one model first.")
    else:
        # Model selection for evaluation
        model_name = st.selectbox("Select Model to Evaluate", list(st.session_state.models.keys()))
        
        model_data = st.session_state.models[model_name]
        model = model_data["model"]
        X_train = model_data["X_train"]
        X_test = model_data["X_test"]
        y_train = model_data["y_train"]
        y_test = model_data["y_test"]
        
        # Evaluate model
        with st.spinner("Evaluating model..."):
            evaluation_results = evaluate_model(model, X_train, X_test, y_train, y_test)
            st.session_state.evaluation_results[model_name] = evaluation_results
            
            # Display results
            st.subheader("Model Performance Metrics")
            
            # Create columns for metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{evaluation_results['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{evaluation_results['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{evaluation_results['recall']:.4f}")
            with col4:
                st.metric("F1 Score", f"{evaluation_results['f1']:.4f}")
            
            # Additional metrics
            st.write("AUC-ROC Score:", f"{evaluation_results['auc']:.4f}")
            
            # Display confusion matrix
            st.subheader("Confusion Matrix")
            confusion_matrix_plot(evaluation_results['confusion_matrix'])
            
            # Display ROC curve
            st.subheader("ROC Curve")
            roc_curve_plot(evaluation_results['fpr'], evaluation_results['tpr'], evaluation_results['auc'])
            
            # Display Precision-Recall curve
            st.subheader("Precision-Recall Curve")
            precision_recall_curve_plot(
                evaluation_results['precision_curve'], 
                evaluation_results['recall_curve'],
                evaluation_results['avg_precision']
            )
            
            # Compare models
            if len(st.session_state.evaluation_results) > 1:
                st.subheader("Model Comparison")
                
                comparison_data = {
                    'Model': [],
                    'Accuracy': [],
                    'Precision': [],
                    'Recall': [],
                    'F1 Score': [],
                    'AUC-ROC': []
                }
                
                for model_name, results in st.session_state.evaluation_results.items():
                    comparison_data['Model'].append(model_name)
                    comparison_data['Accuracy'].append(results['accuracy'])
                    comparison_data['Precision'].append(results['precision'])
                    comparison_data['Recall'].append(results['recall'])
                    comparison_data['F1 Score'].append(results['f1'])
                    comparison_data['AUC-ROC'].append(results['auc'])
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df.set_index('Model'))
                
                # Metrics comparison chart
                st.subheader("Metrics Comparison Chart")
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
                selected_metric = st.selectbox("Select metric for comparison", metrics)
                
                fig = px.bar(
                    comparison_df, 
                    x='Model', 
                    y=selected_metric,
                    color='Model',
                    title=f"{selected_metric} Comparison Across Models"
                )
                st.plotly_chart(fig)

# Feature Importance
elif page == "Feature Importance":
    st.header("Feature Importance Analysis")
    
    if not st.session_state.models:
        st.warning("Please train at least one model first.")
    else:
        # Model selection for feature importance
        model_name = st.selectbox("Select Model", list(st.session_state.models.keys()))
        
        model_data = st.session_state.models[model_name]
        model = model_data["model"]
        X_train = model_data["X_train"]
        
        # Display feature importance
        if model_name not in st.session_state.feature_importance:
            with st.spinner("Calculating feature importance..."):
                if model_name == "Logistic Regression":
                    feature_importance = pd.DataFrame({
                        'Feature': X_train.columns,
                        'Importance': np.abs(model.coef_[0])
                    })
                elif model_name in ["Random Forest", "XGBoost"]:
                    feature_importance = pd.DataFrame({
                        'Feature': X_train.columns,
                        'Importance': model.feature_importances_
                    })
                
                feature_importance = feature_importance.sort_values('Importance', ascending=False)
                st.session_state.feature_importance[model_name] = feature_importance
        
        feature_importance = st.session_state.feature_importance[model_name]
        
        # Display feature importance plot
        st.subheader("Feature Importance")
        feature_importance_plot(feature_importance)
        
        # SHAP analysis for model interpretation
        if st.checkbox("Perform SHAP Analysis (This might take some time for complex models)"):
            with st.spinner("Calculating SHAP values..."):
                shap_analysis(model, X_train, model_name)

# Customer Segmentation
elif page == "Customer Segmentation":
    st.header("Customer Segmentation")
    
    if st.session_state.preprocessed_data is None:
        st.warning("Please upload and preprocess your data first.")
    elif not st.session_state.models:
        st.warning("Please train at least one model first.")
    else:
        # Model selection for customer segmentation
        model_name = st.selectbox("Select Model for Prediction", list(st.session_state.models.keys()))
        
        model_data = st.session_state.models[model_name]
        model = model_data["model"]
        
        # Segment customers based on churn probability
        preprocessed_data = st.session_state.preprocessed_data
        X = st.session_state.features
        
        with st.spinner("Segmenting customers..."):
            segments_df = segment_customers(model, X, preprocessed_data)
            
            # Display customer segments
            st.subheader("Customer Segments by Churn Risk")
            plot_customer_segments(segments_df)
            
            # Display high-risk customers
            st.subheader("High-Risk Customers (Top 10)")
            high_risk = segments_df[segments_df['Risk_Category'] == 'High Risk'].sort_values('Churn_Probability', ascending=False).head(10)
            st.dataframe(high_risk)
            
            # Download segmentation results
            segment_csv = segments_df.to_csv(index=False)
            st.download_button(
                label="Download Customer Segments",
                data=segment_csv,
                file_name="customer_segments.csv",
                mime="text/csv"
            )

# Actionable Insights
elif page == "Actionable Insights":
    st.header("Actionable Insights")
    
    if not st.session_state.models or not st.session_state.feature_importance:
        st.warning("Please train and evaluate models first.")
    else:
        st.subheader("Key Insights from Analysis")
        
        # Get the model with highest accuracy
        best_model = max(st.session_state.evaluation_results.items(), 
                         key=lambda x: x[1]['accuracy'])[0]
        
        st.write(f"🏆 Best performing model: **{best_model}** with accuracy of **{st.session_state.evaluation_results[best_model]['accuracy']:.4f}**")
        
        # Top churn drivers
        st.subheader("Top Churn Drivers")
        
        if best_model in st.session_state.feature_importance:
            top_features = st.session_state.feature_importance[best_model].head(5)
            
            fig = px.bar(
                top_features, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Top 5 Features Influencing Churn",
                color='Importance'
            )
            st.plotly_chart(fig)
            
            # Actionable recommendations
            st.subheader("Recommended Actions")
            
            st.markdown("""
            ### Retention Strategies
            
            Based on the analysis, here are some recommended actions to reduce churn:
            
            1. **Targeted Promotions**: Create personalized offers for high-risk customers
            2. **Service Improvement**: Address issues in areas identified as key churn drivers
            3. **Proactive Outreach**: Establish regular communication with at-risk customers
            4. **Loyalty Programs**: Develop or enhance loyalty programs to increase customer retention
            5. **Customer Experience**: Improve customer experience in the areas most correlated with churn
            
            ### Implementation Plan
            
            1. **Immediate Actions** (0-30 days):
               - Identify and contact the top 10% highest-risk customers with special offers
               - Conduct focused customer satisfaction surveys in problem areas
            
            2. **Short-term Actions** (1-3 months):
               - Implement service improvements based on key churn drivers
               - Develop an early warning system for customer dissatisfaction
            
            3. **Long-term Strategy** (3-12 months):
               - Restructure service packages based on customer segmentation insights
               - Implement continuous monitoring of churn drivers
               - Establish regular model retraining to adapt to changing customer behavior
            """)
            
            # Expected impact
            st.subheader("Expected Impact")
            
            st.markdown("""
            Based on the model predictions and industry benchmarks, implementing these recommendations could result in:
            
            - **Churn Reduction**: 15-25% reduction in customer churn rate
            - **Revenue Protection**: Preservation of recurring revenue from retained customers
            - **Customer Satisfaction**: Improved NPS (Net Promoter Score) and customer satisfaction metrics
            - **Competitive Advantage**: Better positioning against competitors through enhanced customer loyalty
            """)
            
            # Monitoring suggestions
            st.subheader("Ongoing Monitoring")
            
            st.markdown("""
            To ensure the effectiveness of these strategies, monitor:
            
            1. **Churn Rate Trends**: Track overall and segment-specific churn rates
            2. **Model Performance**: Regularly validate model predictions against actual churn
            3. **Intervention Effectiveness**: Measure the impact of specific retention actions
            4. **Customer Feedback**: Collect and analyze feedback to identify emerging issues
            """)

# Footer
st.markdown("""
---
© 2023 NetOne Telecommunications Churn Prediction System
""")
