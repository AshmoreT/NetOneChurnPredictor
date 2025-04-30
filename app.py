import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import io
import os
import json
import pickle
import datetime

from utils.data_preprocessing import preprocess_data, handle_missing_values, encode_categorical_features
from utils.exploratory_data_analysis import plot_churn_distribution, plot_feature_correlation, plot_categorical_features, plot_numerical_features
from utils.model_training import train_logistic_regression, train_random_forest, train_xgboost
from utils.model_evaluation import evaluate_model, confusion_matrix_plot, roc_curve_plot, precision_recall_curve_plot
from utils.model_interpretation import feature_importance_plot, shap_analysis
from utils.customer_segmentation import segment_customers, plot_customer_segments
from utils.authentication import render_login_page, check_authentication, logout
from utils.database import get_user_by_username, add_dataset, add_model, add_prediction, get_user_datasets, get_user_models

# Set page configuration
st.set_page_config(
    page_title="NetOne Churn Prediction System",
    page_icon="📊",
    layout="wide"
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

# Check if user is authenticated
if not check_authentication():
    render_login_page()
    # Exit early to prevent showing application content
    st.stop()

# User is authenticated, show main application
# Get screen width for responsive design
# Application title and introduction
st.title("NetOne Telecommunications Churn Prediction System")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Home Dashboard", "Data Upload & Preprocessing", "Exploratory Data Analysis", "Model Training", 
     "Model Evaluation", "Feature Importance", "Customer Segmentation", "Actionable Insights"]
)

# Professional home page and dashboard
if page == "Home Dashboard":
    st.header("Welcome to NetOne Churn Prediction System")
    
    # Create a professional layout with cards/sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("About the System")
        st.markdown("""
        <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;">
        <h4>Predict Customer Churn with Advanced Analytics</h4>
        <p>NetOne's Churn Prediction System leverages advanced machine learning algorithms to identify at-risk customers 
        before they leave. Our system analyzes customer behavior patterns, service usage, and historical data to provide 
        actionable insights for reducing customer attrition.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;margin-top:20px;">
        <h4>Works 100% Offline</h4>
        <p>This system is designed to operate entirely offline with no external dependencies. 
        Your sensitive customer data never leaves your secure environment, ensuring complete 
        privacy and compliance with data protection regulations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Key Features")
        st.markdown("""
        <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;">
        <h4>Complete ML Pipeline</h4>
        <ul>
            <li><strong>Data Preprocessing:</strong> Clean, transform and prepare your data</li>
            <li><strong>Exploratory Analysis:</strong> Visualize and understand patterns</li>
            <li><strong>Model Training:</strong> Build powerful predictive models</li>
            <li><strong>Customer Segmentation:</strong> Identify distinct customer groups</li>
            <li><strong>Actionable Insights:</strong> Get recommendations to reduce churn</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;margin-top:20px;">
        <h4>Quick Start</h4>
        <p>Navigate to <strong>Data Upload & Preprocessing</strong> in the sidebar to begin. 
        Upload your customer dataset in CSV or Excel format, then follow the guided workflow to 
        generate predictions and insights.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # System usage stats and charts
    st.subheader("System Overview")
    
    # Create some statistics about current state
    try:
        user = get_user_by_username(st.session_state.username)
        if user and hasattr(user, 'id'):
            datasets = get_user_datasets(user.id)
            models = get_user_models(user.id)
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(label="Saved Datasets", value=len(datasets) if datasets else 0)
            
            with col2:
                st.metric(label="Trained Models", value=len(models) if models else 0)
            
            with col3:
                # Calculate average model accuracy if models exist
                if models and len(models) > 0:
                    avg_accuracy = sum(model.accuracy for model in models) / len(models)
                    st.metric(label="Avg. Model Accuracy", value=f"{avg_accuracy:.2%}")
                else:
                    st.metric(label="Avg. Model Accuracy", value="N/A")
            
            with col4:
                # Show when the last model was created
                if models and len(models) > 0:
                    latest_model = max(models, key=lambda x: x.created_at)
                    days_ago = (datetime.datetime.utcnow() - latest_model.created_at).days
                    st.metric(label="Last Model Created", value=f"{days_ago} days ago" if days_ago > 0 else "Today")
                else:
                    st.metric(label="Last Model Created", value="N/A")
            
            # Show recent activity if available
            if models and len(models) > 0:
                st.subheader("Recent Models")
                recent_models_df = pd.DataFrame({
                    "Model Name": [model.name for model in models[:3]],
                    "Type": [model.model_type for model in models[:3]],
                    "Accuracy": [f"{model.accuracy:.2%}" for model in models[:3]],
                    "Created": [model.created_at.strftime("%Y-%m-%d") for model in models[:3]]
                })
                st.dataframe(recent_models_df)
        
        else:
            st.info("No user data found. Start by uploading a dataset and training models.")
    
    except Exception as e:
        st.warning(f"Could not load system statistics: {str(e)}")
        
    # Help section
    st.subheader("Need Help?")
    with st.expander("How to use this system"):
        st.markdown("""
        1. **Upload your data**: Go to the Data Upload & Preprocessing page and upload your customer dataset
        2. **Explore your data**: Use the Exploratory Data Analysis page to understand patterns and relationships
        3. **Train models**: Select the best features and train different models to predict churn
        4. **Evaluate performance**: Compare models and select the best performing one
        5. **Generate insights**: Use the Customer Segmentation and Actionable Insights pages to develop retention strategies
        """)
    
    with st.expander("About the models"):
        st.markdown("""
        This system offers three powerful machine learning algorithms:
        
        - **Logistic Regression**: Simple, interpretable model good for understanding feature importance
        - **Random Forest**: Ensemble method that handles non-linear relationships well
        - **XGBoost**: Advanced gradient boosting algorithm for maximum prediction accuracy
        
        Each model has strengths and weaknesses. We recommend comparing their performance on your specific dataset.
        """)
else:
    # Display user info and logout button in sidebar (outside the page conditional)
    if page != "Home Dashboard":  # Only show the welcome message if not on home page
        st.sidebar.title(f"Welcome, {st.session_state.username}")
        st.sidebar.write(f"Role: {st.session_state.role}")
    else:
        # If on home page, just show the logout button without duplicating welcome
        st.sidebar.title("User Options")

# Add logout button
if logout():
    st.rerun()

# Show user's saved data and models
with st.sidebar.expander("Your Saved Data"):
    try:
        user = get_user_by_username(st.session_state.username)
        if user and hasattr(user, 'id'):
            # User exists and has an ID
            datasets = get_user_datasets(user.id)
            if datasets:
                st.write("Your Datasets:")
                for dataset in datasets:
                    st.write(f"- {dataset.name} ({dataset.rows} rows)")
            else:
                st.write("No datasets saved yet.")
                
            models = get_user_models(user.id)
            if models:
                st.write("Your Models:")
                for model in models:
                    st.write(f"- {model.name} ({model.model_type})")
                    st.write(f"  Accuracy: {model.accuracy:.4f}")
            else:
                st.write("No models saved yet.")
        else:
            st.write("User profile not found in database.")
    except Exception as e:
        st.write(f"Error loading saved data: {str(e)}")

# Note: Navigation is already handled at the top of the file

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
                    
                    # Save dataset to database
                    try:
                        # Get user info
                        user = get_user_by_username(st.session_state.username)
                        
                        if user and hasattr(user, 'id'):
                            # Define a filename for the data
                            dataset_name = f"churn_dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            dataset_path = f"datasets/{dataset_name}"
                            
                            # Create the directory if it doesn't exist
                            os.makedirs("datasets", exist_ok=True)
                            
                            # Save the dataset to a file
                            data.to_csv(dataset_path, index=False)
                            
                            # Save to database
                            dataset_success, dataset_id = add_dataset(
                                name=dataset_name,
                                description=f"Churn dataset uploaded on {datetime.datetime.now().strftime('%Y-%m-%d')}",
                                user_id=user.id,
                                file_path=dataset_path,
                                rows=data.shape[0],
                                columns=data.shape[1],
                                target_column=target_column
                            )
                            
                            if dataset_success:
                                st.session_state.dataset_id = dataset_id
                                st.success("Data preprocessing completed and dataset saved to database!")
                            else:
                                st.warning("Data preprocessing completed but failed to save to database.")
                        else:
                            st.success("Data preprocessing completed successfully!")
                            st.warning("Dataset not saved to database: User profile not found.")
                    except Exception as e:
                        st.success("Data preprocessing completed successfully!")
                        st.warning(f"Database operation failed: {str(e)}")
                    
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
                        
                        # Save model to file and database
                        try:
                            # Get user info
                            user = get_user_by_username(st.session_state.username)
                            
                            # Directory for models
                            os.makedirs("models", exist_ok=True)
                            
                            # Model file path
                            model_file = f"models/logistic_regression_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                            
                            # Save model to file
                            with open(model_file, 'wb') as f:
                                pickle.dump(model, f)
                            
                            # Preliminary evaluation metrics
                            evaluation_results = evaluate_model(model, X_train, X_test, y_train, y_test)
                            
                            # Save model to database if user exists
                            if user and hasattr(user, 'id') and 'dataset_id' in st.session_state:
                                dataset_id = st.session_state.dataset_id
                                
                                model_success, model_id = add_model(
                                    name=f"Logistic Regression Model ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})",
                                    description=f"C={C}, solver={solver}, max_iter={max_iter}",
                                    user_id=user.id,
                                    dataset_id=dataset_id,
                                    model_type="Logistic Regression",
                                    accuracy=evaluation_results['accuracy'],
                                    precision=evaluation_results['precision'],
                                    recall=evaluation_results['recall'],
                                    f1_score=evaluation_results['f1'],
                                    auc_score=evaluation_results['auc'],
                                    file_path=model_file
                                )
                                
                                if model_success:
                                    st.session_state.model_id = model_id
                                    st.success("Logistic Regression model trained successfully and saved to database!")
                                else:
                                    st.success("Logistic Regression model trained successfully!")
                                    st.warning("Failed to save model to database.")
                            elif not user or not hasattr(user, 'id'):
                                st.success("Logistic Regression model trained successfully and saved to file!")
                                st.warning("Model not saved to database: User profile not found.")
                            else:
                                st.success("Logistic Regression model trained successfully and saved to file!")
                                st.warning("Model not saved to database: No dataset ID found.")
                        except Exception as e:
                            st.success("Logistic Regression model trained successfully!")
                            st.warning(f"Failed to save model: {str(e)}")
            
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
                        
                        # Save model to file and database
                        try:
                            # Get user info
                            user = get_user_by_username(st.session_state.username)
                            
                            # Directory for models
                            os.makedirs("models", exist_ok=True)
                            
                            # Model file path
                            model_file = f"models/random_forest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                            
                            # Save model to file
                            with open(model_file, 'wb') as f:
                                pickle.dump(model, f)
                            
                            # Preliminary evaluation metrics
                            evaluation_results = evaluate_model(model, X_train, X_test, y_train, y_test)
                            
                            # Save model to database if user exists
                            if user and hasattr(user, 'id') and 'dataset_id' in st.session_state:
                                dataset_id = st.session_state.dataset_id
                                
                                model_success, model_id = add_model(
                                    name=f"Random Forest Model ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})",
                                    description=f"n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}",
                                    user_id=user.id,
                                    dataset_id=dataset_id,
                                    model_type="Random Forest",
                                    accuracy=evaluation_results['accuracy'],
                                    precision=evaluation_results['precision'],
                                    recall=evaluation_results['recall'],
                                    f1_score=evaluation_results['f1'],
                                    auc_score=evaluation_results['auc'],
                                    file_path=model_file
                                )
                                
                                if model_success:
                                    st.session_state.model_id = model_id
                                    st.success("Random Forest model trained successfully and saved to database!")
                                else:
                                    st.success("Random Forest model trained successfully!")
                                    st.warning("Failed to save model to database.")
                            elif not user or not hasattr(user, 'id'):
                                st.success("Random Forest model trained successfully and saved to file!")
                                st.warning("Model not saved to database: User profile not found.")
                            else:
                                st.success("Random Forest model trained successfully and saved to file!")
                                st.warning("Model not saved to database: No dataset ID found.")
                        except Exception as e:
                            st.success("Random Forest model trained successfully!")
                            st.warning(f"Failed to save model: {str(e)}")
            
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
                        
                        # Save model to file and database
                        try:
                            # Get user info
                            user = get_user_by_username(st.session_state.username)
                            
                            # Directory for models
                            os.makedirs("models", exist_ok=True)
                            
                            # Model file path
                            model_file = f"models/xgboost_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                            
                            # Save model to file
                            with open(model_file, 'wb') as f:
                                pickle.dump(model, f)
                            
                            # Preliminary evaluation metrics
                            evaluation_results = evaluate_model(model, X_train, X_test, y_train, y_test)
                            
                            # Save model to database if user exists
                            if user and hasattr(user, 'id') and 'dataset_id' in st.session_state:
                                dataset_id = st.session_state.dataset_id
                                
                                model_success, model_id = add_model(
                                    name=f"XGBoost Model ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})",
                                    description=f"n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}, subsample={subsample}",
                                    user_id=user.id,
                                    dataset_id=dataset_id,
                                    model_type="XGBoost",
                                    accuracy=evaluation_results['accuracy'],
                                    precision=evaluation_results['precision'],
                                    recall=evaluation_results['recall'],
                                    f1_score=evaluation_results['f1'],
                                    auc_score=evaluation_results['auc'],
                                    file_path=model_file
                                )
                                
                                if model_success:
                                    st.session_state.model_id = model_id
                                    st.success("XGBoost model trained successfully and saved to database!")
                                else:
                                    st.success("XGBoost model trained successfully!")
                                    st.warning("Failed to save model to database.")
                            elif not user or not hasattr(user, 'id'):
                                st.success("XGBoost model trained successfully and saved to file!")
                                st.warning("Model not saved to database: User profile not found.")
                            else:
                                st.success("XGBoost model trained successfully and saved to file!")
                                st.warning("Model not saved to database: No dataset ID found.")
                        except Exception as e:
                            st.success("XGBoost model trained successfully!")
                            st.warning(f"Failed to save model: {str(e)}")
        
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
            
            # Save predictions to database
            if 'model_id' in st.session_state:
                try:
                    model_id = st.session_state.model_id
                    
                    # Save high risk customers to the database
                    for idx, row in high_risk.iterrows():
                        customer_id = str(row.get('CustomerID', f"CUST_{idx}"))
                        features = json.dumps({col: str(row[col]) for col in X.columns if col in row})
                        prediction = float(row['Churn_Probability'])
                        is_churn = bool(row['Risk_Category'] == 'High Risk')
                        
                        # Add prediction to database
                        add_prediction(
                            model_id=model_id,
                            customer_id=customer_id,
                            features=features,
                            prediction=prediction,
                            is_churn=is_churn
                        )
                    
                    st.success(f"Saved {len(high_risk)} high-risk customer predictions to database")
                except Exception as e:
                    st.warning(f"Failed to save predictions to database: {str(e)}")
            
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
