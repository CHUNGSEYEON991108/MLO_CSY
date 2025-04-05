# mlops_csy/loader.py
import pandas as pd

def load_data(path='./data/train.csv'):
    try:
        df = pd.read_csv(path)
        print("✅ 데이터 로딩 완료!")
        return df
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {path}")
        raise
