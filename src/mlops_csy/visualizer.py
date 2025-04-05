import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(data, save_path=None):
    """
    Create visualizations for the data
    
    Args:
        data (pandas.DataFrame): Input data
        save_path (str, optional): Path to save the visualizations
    """
    # Set up the matplotlib figure
    plt.figure(figsize=(15, 10))
    
    # Create subplots for numeric columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    n_cols = len(numeric_cols)
    
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2, (n_cols + 1) // 2, i)
        sns.histplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close() 