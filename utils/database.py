import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
import hashlib

# Get database URL from environment variable
DATABASE_URL = os.environ.get('DATABASE_URL')

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Create a base class for declarative models
Base = declarative_base()

# Define User model
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(100), nullable=False)  # Stores hashed password
    email = Column(String(100), unique=True, nullable=False)
    role = Column(String(20), default='user')
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationship with datasets
    datasets = relationship("Dataset", back_populates="user")
    # Relationship with models
    models = relationship("Model", back_populates="user")
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}', role='{self.role}')>"

# Define Dataset model
class Dataset(Base):
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    user_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    file_path = Column(String(255))  # Path to stored dataset file
    rows = Column(Integer)
    columns = Column(Integer)
    target_column = Column(String(100))
    
    # Relationship with user
    user = relationship("User", back_populates="datasets")
    # Relationship with models
    models = relationship("Model", back_populates="dataset")
    
    def __repr__(self):
        return f"<Dataset(name='{self.name}', rows={self.rows}, columns={self.columns})>"

# Define Model model
class Model(Base):
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    user_id = Column(Integer, ForeignKey('users.id'))
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    model_type = Column(String(50))  # Logistic Regression, Random Forest, XGBoost
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_score = Column(Float)
    file_path = Column(String(255))  # Path to stored model file
    
    # Relationship with user
    user = relationship("User", back_populates="models")
    # Relationship with dataset
    dataset = relationship("Dataset", back_populates="models")
    # Relationship with predictions
    predictions = relationship("Prediction", back_populates="model")
    
    def __repr__(self):
        return f"<Model(name='{self.name}', type='{self.model_type}', accuracy={self.accuracy})>"

# Define Prediction model
class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('models.id'))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    customer_id = Column(String(100))
    features = Column(Text)  # JSON string of features
    prediction = Column(Float)  # Churn probability
    is_churn = Column(Boolean)  # True if customer is predicted to churn
    
    # Relationship with model
    model = relationship("Model", back_populates="predictions")
    
    def __repr__(self):
        return f"<Prediction(customer_id='{self.customer_id}', prediction={self.prediction}, is_churn={self.is_churn})>"

# Create tables
def init_db():
    Base.metadata.create_all(engine)

# Add a user
def add_user(username, password, email, role='user'):
    # Hash the password
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    session = Session()
    try:
        # Check if user already exists
        existing_user = session.query(User).filter(User.username == username).first()
        if existing_user:
            session.close()
            return False, "Username already exists"
        
        # Check if email already exists
        existing_email = session.query(User).filter(User.email == email).first()
        if existing_email:
            session.close()
            return False, "Email already exists"
        
        # Create new user
        new_user = User(username=username, password=hashed_password, email=email, role=role)
        session.add(new_user)
        session.commit()
        return True, "User added successfully"
    except Exception as e:
        session.rollback()
        return False, str(e)
    finally:
        session.close()

# Authenticate a user
def authenticate_user(username, password):
    # Hash the password
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    session = Session()
    try:
        # Check if user exists and password matches
        user = session.query(User).filter(User.username == username, User.password == hashed_password).first()
        if user:
            return True, user.role
        return False, None
    except Exception as e:
        return False, str(e)
    finally:
        session.close()

# Add a dataset
def add_dataset(name, description, user_id, file_path, rows, columns, target_column):
    session = Session()
    try:
        # Create new dataset
        new_dataset = Dataset(
            name=name,
            description=description,
            user_id=user_id,
            file_path=file_path,
            rows=rows,
            columns=columns,
            target_column=target_column
        )
        session.add(new_dataset)
        session.commit()
        return True, new_dataset.id
    except Exception as e:
        session.rollback()
        return False, str(e)
    finally:
        session.close()

# Get datasets for a user
def get_user_datasets(user_id):
    session = Session()
    try:
        datasets = session.query(Dataset).filter(Dataset.user_id == user_id).all()
        return datasets
    except Exception as e:
        return []
    finally:
        session.close()

# Add a model
def add_model(name, description, user_id, dataset_id, model_type, accuracy, precision, recall, f1_score, auc_score, file_path):
    session = Session()
    try:
        # Create new model
        new_model = Model(
            name=name,
            description=description,
            user_id=user_id,
            dataset_id=dataset_id,
            model_type=model_type,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            auc_score=auc_score,
            file_path=file_path
        )
        session.add(new_model)
        session.commit()
        return True, new_model.id
    except Exception as e:
        session.rollback()
        return False, str(e)
    finally:
        session.close()

# Get models for a user
def get_user_models(user_id):
    session = Session()
    try:
        models = session.query(Model).filter(Model.user_id == user_id).all()
        return models
    except Exception as e:
        return []
    finally:
        session.close()

# Add a prediction
def add_prediction(model_id, customer_id, features, prediction, is_churn):
    session = Session()
    try:
        # Create new prediction
        new_prediction = Prediction(
            model_id=model_id,
            customer_id=customer_id,
            features=features,
            prediction=prediction,
            is_churn=is_churn
        )
        session.add(new_prediction)
        session.commit()
        return True, new_prediction.id
    except Exception as e:
        session.rollback()
        return False, str(e)
    finally:
        session.close()

# Get predictions for a model
def get_model_predictions(model_id):
    session = Session()
    try:
        predictions = session.query(Prediction).filter(Prediction.model_id == model_id).all()
        return predictions
    except Exception as e:
        return []
    finally:
        session.close()

# Get user by username
def get_user_by_username(username):
    session = Session()
    try:
        user = session.query(User).filter(User.username == username).first()
        return user
    except Exception as e:
        return None
    finally:
        session.close()

# Initialize the database
init_db()