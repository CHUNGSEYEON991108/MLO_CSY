from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train a random forest classifier
    
    Args:
        X (array-like): Features
        y (array-like): Target variable
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (trained model, X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def predict(model, X):
    """
    Make predictions using the trained model
    
    Args:
        model: Trained model
        X (array-like): Features to predict
        
    Returns:
        tuple: (Predicted labels, Predicted probabilities)
    """
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    return y_pred, y_pred_proba 