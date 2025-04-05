# mlops_csy/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def count_unique_values(df, col):
    print(f"\n🔹 '{col}' 고유값 분포:")
    print(df[col].value_counts())
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f"{col} 분포")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_distribution(df, col):
    print(f"\n📈 '{col}' 수치 분포:")
    sns.histplot(df[col].dropna(), bins=30, kde=True)
    plt.title(f"{col} 분포")
    plt.xlabel(col)
    plt.ylabel('빈도수')
    plt.tight_layout()
    plt.show()
