# mlops_csy/model_utils.py
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def train_model(df, target_col='채무 불이행 여부'):
    # 특징(X)과 목표 변수(y) 정의
    X = df.drop(columns=[target_col, 'UID'])  # '채무 불이행 여부'와 'UID'를 제외한 데이터 사용
    y = df[target_col]  # '채무 불이행 여부'를 목표 변수로 설정

    # 범주형 변수 처리
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category').cat.codes  # 범주형 데이터를 코드로 변환

    # 수치형 변수 스케일링 추가
    scaler = StandardScaler()
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    # 학습 데이터와 검증 데이터로 나누기
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 개별 모델 정의 (성능 개선을 위한 하이퍼파라미터 조정)
    rf_model = RandomForestClassifier(
        n_estimators=100,    # 트리 수 증가
        max_depth=15,        # 트리 깊이 증가
        min_samples_split=5, # 분할을 위한 최소 샘플 수
        min_samples_leaf=2,  # 리프 노드의 최소 샘플 수
        n_jobs=-1,          # 모든 CPU 코어 사용
        random_state=42
    )
    
    lr_model = LogisticRegression(
        C=0.1,              # 규제 강도 조정
        max_iter=1000,      # 최대 반복 횟수
        class_weight='balanced', # 클래스 불균형 처리
        n_jobs=-1,          # 모든 CPU 코어 사용
        random_state=42
    )

    # 앙상블 모델 (VotingClassifier)
    ensemble_model = VotingClassifier(
        estimators=[('rf', rf_model), ('lr', lr_model)],
        voting='soft',  # 확률 기반 투표 방식 사용
        n_jobs=-1      # 병렬 처리 활성화
    )

    # 모델 학습
    print("\n🔄 모델 학습 중...")
    ensemble_model.fit(X_train, y_train)

    # 검증 정확도 출력
    y_pred = ensemble_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"\n✅ 검증 정확도 (앙상블 모델): {acc:.4f}")
    print("\n📋 분류 리포트:\n", classification_report(y_val, y_pred))
    
    return ensemble_model

def predict_and_save(model, test_df, submission_path):
    # 테스트 데이터에서 'UID' 열 제외하고 예측
    X_test = test_df.drop(columns=['UID'])
    
    # 범주형 변수 처리
    for col in X_test.select_dtypes(include='object').columns:
        X_test[col] = X_test[col].astype('category').cat.codes  # 범주형 데이터를 코드로 변환

    # 예측 확률 (채무 불이행 확률)
    print("\n🔄 예측 중...")
    preds = model.predict_proba(X_test)[:, 1]  # 양성 클래스(채무 불이행 확률) 예측

    # 예측 결과를 submission 파일로 저장
    submission = pd.DataFrame({
        'UID': test_df['UID'],
        '채무 불이행 확률': preds
    })
    
    # 결과 저장
    submission.to_csv(submission_path, index=False)
    print(f"\n✅ 예측 결과가 {submission_path}에 저장되었습니다.")
