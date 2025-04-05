# mlops_csy/model_utils.py
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .metrics import calculate_roc_auc, plot_roc_curve, print_model_performance

def preprocess_data(df, is_training=True):
    """데이터 전처리를 수행합니다."""
    # UID 제거
    if 'UID' in df.columns:
        df = df.drop('UID', axis=1)
    
    # 목표 변수 분리
    X = df.drop('채무 불이행 여부', axis=1) if is_training else df
    y = df['채무 불이행 여부'] if is_training else None
    
    # 범주형 변수 처리
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category').cat.codes
    
    return X, y

def create_model():
    """모델 파이프라인을 생성합니다."""
    # 개별 모델 정의
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    lr = LogisticRegression(
        C=0.1,
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # 파이프라인 생성
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', rf)  # 초기 모델로 RandomForest 사용
    ])
    
    return pipeline

def train_model(df, target_col='채무 불이행 여부'):
    """모델을 학습하고 평가합니다."""
    print("\n🔄 데이터 전처리 중...")
    X, y = preprocess_data(df)
    
    # 학습/검증 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 모델 생성 및 학습
    print("\n🔄 모델 학습 중...")
    model = create_model()
    model.fit(X_train, y_train)
    
    # 교차 검증 수행
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=5, scoring='roc_auc'
    )
    print(f"\n📊 교차 검증 ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 검증 세트에서 성능 평가
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    roc_auc = print_model_performance(y_val, val_pred_proba)
    
    # ROC 곡선 그리기 및 저장
    plot_roc_curve(y_val, val_pred_proba, save_path='output/roc_curve.png')
    
    return model

def predict_and_save(model, test_df, submission_path):
    """테스트 데이터에 대한 예측을 수행하고 저장합니다."""
    print("\n🔄 예측 중...")
    
    # 데이터 전처리
    X_test, _ = preprocess_data(test_df, is_training=False)
    
    # 예측 확률 계산
    pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 제출 파일 생성
    submission = pd.DataFrame({
        'UID': test_df['UID'],
        '채무 불이행 확률': pred_proba
    })
    
    # 결과 저장
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission.to_csv(submission_path, index=False)
    print(f"\n✅ 예측 결과가 {submission_path}에 저장되었습니다.")
