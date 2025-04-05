import os
import pandas as pd

def predict_and_save(model, test_df, submission_path):
    # 테스트 데이터에서 'UID' 열 제외하고 예측
    X_test = test_df.drop(columns=['UID'])
    
    # 범주형 변수 처리
    for col in X_test.select_dtypes(include='object').columns:
        X_test[col] = X_test[col].astype('category').cat.codes  # 범주형 데이터를 코드로 변환

    # 예측 확률 (채무 불이행 확률)
    preds = model.predict_proba(X_test)[:, 1]  # 양성 클래스(채무 불이행 확률) 예측

    # 예측 결과를 submission 파일로 저장
    submission = pd.DataFrame({
        'UID': test_df['UID'],
        '채무 불이행 확률': preds
    })
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission.to_csv(submission_path, index=False)
    print(f"📄 예측 결과 저장 완료: {submission_path}")
