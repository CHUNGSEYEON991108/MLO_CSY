import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import font_manager, rc

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우의 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

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

def count_unique_values(data, column, save_path=None):
    """
    Count and display unique values in a column
    
    Args:
        data (pandas.DataFrame): Input data
        column (str): Column name to analyze
        save_path (str, optional): Path to save the visualization
    """
    value_counts = data[column].value_counts()
    print(f"\n🔹 '{column}' 고유값 분포:")
    print(value_counts)
    
    # 시각화 추가
    plt.figure(figsize=(12, 6))
    value_counts.plot(kind='bar')
    plt.title(f"'{column}' 분포")
    plt.xlabel(column)
    plt.ylabel("건수")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_distribution(data, column, save_path=None):
    """
    Plot distribution of a numeric column
    
    Args:
        data (pandas.DataFrame): Input data
        column (str): Column name to plot
        save_path (str, optional): Path to save the visualization
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(data[column], kde=True)
    plt.title(f"📈 '{column}' 수치 분포")
    plt.xlabel(column)
    plt.ylabel("건수")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close() 