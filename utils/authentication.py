import streamlit as st
import pandas as pd
import hashlib
import os
import json

# Path to store user data
USER_DB_PATH = "users.json"

def load_users():
    """
    Load users from the JSON file
    
    Returns:
    --------
    dict
        Dictionary with username as key and user info as value
    """
    if os.path.exists(USER_DB_PATH):
        with open(USER_DB_PATH, "r") as f:
            return json.load(f)
    else:
        return {}

def save_users(users):
    """
    Save users to the JSON file
    
    Parameters:
    -----------
    users : dict
        Dictionary with username as key and user info as value
    """
    with open(USER_DB_PATH, "w") as f:
        json.dump(users, f)

def hash_password(password):
    """
    Hash a password for storing
    
    Parameters:
    -----------
    password : str
        Password to hash
        
    Returns:
    --------
    str
        Hashed password
    """
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email, role="user"):
    """
    Register a new user
    
    Parameters:
    -----------
    username : str
        Username for the new user
    password : str
        Password for the new user
    email : str
        Email for the new user
    role : str
        Role for the new user ('user' or 'admin')
        
    Returns:
    --------
    bool
        True if registration is successful, False otherwise
    """
    users = load_users()
    
    # Check if username already exists
    if username in users:
        return False
    
    # Create new user
    users[username] = {
        "password": hash_password(password),
        "email": email,
        "role": role
    }
    
    # Save updated users
    save_users(users)
    return True

def authenticate(username, password):
    """
    Authenticate a user
    
    Parameters:
    -----------
    username : str
        Username to authenticate
    password : str
        Password to authenticate
        
    Returns:
    --------
    bool
        True if authentication is successful, False otherwise
    """
    users = load_users()
    
    # Check if username exists and password matches
    if username in users and users[username]["password"] == hash_password(password):
        return True
    
    return False

def get_user_role(username):
    """
    Get the role of a user
    
    Parameters:
    -----------
    username : str
        Username to get role for
        
    Returns:
    --------
    str
        Role of the user
    """
    users = load_users()
    
    if username in users:
        return users[username]["role"]
    
    return None

def login_form():
    """
    Display the login form and handle login logic
    
    Returns:
    --------
    bool
        True if login is successful, False otherwise
    """
    st.subheader("Login")
    
    # Login form
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Login", key="login_button"):
        if authenticate(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.role = get_user_role(username)
            st.success(f"Welcome back, {username}!")
            return True
        else:
            st.error("Invalid username or password")
    
    return False

def registration_form():
    """
    Display the registration form and handle registration logic
    
    Returns:
    --------
    bool
        True if registration is successful, False otherwise
    """
    st.subheader("Register")
    
    # Registration form
    username = st.text_input("Username", key="register_username")
    email = st.text_input("Email", key="register_email")
    password = st.text_input("Password", type="password", key="register_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")
    
    if st.button("Register", key="register_button"):
        # Check if passwords match
        if password != confirm_password:
            st.error("Passwords do not match")
            return False
        
        # Register user
        if register_user(username, password, email):
            st.success("Registration successful! You can now login.")
            return True
        else:
            st.error("Username already exists")
    
    return False

def logout():
    """
    Log out the user
    """
    if st.button("Logout", key="logout_button"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.role = None
        return True
    
    return False

def check_authentication():
    """
    Check if the user is authenticated
    
    Returns:
    --------
    bool
        True if the user is authenticated, False otherwise
    """
    return st.session_state.get('authenticated', False)

def render_login_page():
    """
    Render the login page with login and registration forms
    
    Returns:
    --------
    bool
        True if the user is authenticated, False otherwise
    """
    st.title("NetOne Churn Prediction System")
    st.write("Please login to continue")
    
    # Create tabs for login and registration
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        if login_form():
            return True
    
    with register_tab:
        registration_form()
    
    return False