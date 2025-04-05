import pandas as pd
import numpy as np

def explore_data(data):
    """
    Perform exploratory data analysis
    
    Args:
        data (pandas.DataFrame): Input data
        
    Returns:
        dict: Dictionary containing analysis results
    """
    analysis = {
        'shape': data.shape,
        'columns': list(data.columns),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_summary': data.describe().to_dict(),
        'categorical_summary': {
            col: data[col].value_counts().to_dict()
            for col in data.select_dtypes(include=['object']).columns
        }
    }
    
    return analysis 