# mlops_csy/explorer.py
def explore_data(df):
    print("\n📦 데이터 크기:", df.shape)
    print("\n🧾 컬럼 목록:", df.columns.tolist())
    print("\n🧪 컬럼별 데이터 타입:\n", df.dtypes)
    print("\n❌ 결측치 수:\n", df.isnull().sum())
    print("\n📊 수치형 통계 요약:\n", df.describe())
    print("\n🧩 범주형 통계 요약:\n", df.describe(include='object'))
