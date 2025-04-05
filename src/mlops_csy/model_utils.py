from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import mlflow
import mlflow.sklearn

def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        print(f"ROC 커브가 {save_path}에 저장되었습니다.")
    
    plt.close()
    return roc_auc

def evaluate_model(model, X, y):
    """
    Evaluate model performance using various metrics
    
    Args:
        model: Trained model
        X: Features
        y: True labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': plot_roc_curve(y, y_pred_proba)
    }
    
    return metrics

def train_model(data, output_dir=None, experiment_name="채무불이행예측"):
    """
    Train a random forest classifier with MLflow tracking
    
    Args:
        data (pandas.DataFrame): Training data with features and target
        output_dir (str, optional): Directory to save plots
        experiment_name (str): Name of the MLflow experiment
        
    Returns:
        RandomForestClassifier: Trained model
    """
    # MLflow 실험 설정
    mlflow.set_experiment(experiment_name)
    
    # 특성과 타겟 분리
    X = data.drop(['UID', '채무 불이행 여부'], axis=1)
    y = data['채무 불이행 여부']
    
    # 범주형 변수 처리
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols)
    
    # 학습/검증 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # MLflow로 실험 추적
    with mlflow.start_run():
        # 모델 하이퍼파라미터 설정
        model_params = {
            'n_estimators': 100,
            'random_state': 42
        }
        
        # 실험 파라미터 기록 (모델 및 데이터 분할 파라미터 포함)
        mlflow.log_params({
            **model_params,
            'test_size': 0.2
        })
        
        # 모델 학습
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        
        # 교차 검증 수행
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        print(f"\n📊 교차 검증 ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 검증 데이터로 모델 평가
        metrics = evaluate_model(model, X_val, y_val)
        
        # 메트릭 기록
        mlflow.log_metrics({
            'cv_roc_auc_mean': cv_scores.mean(),
            'cv_roc_auc_std': cv_scores.std(),
            'val_accuracy': metrics['accuracy'],
            'val_precision': metrics['precision'],
            'val_recall': metrics['recall'],
            'val_f1': metrics['f1'],
            'val_roc_auc': metrics['roc_auc']
        })
        
        # ROC 커브 그리기 및 저장
        if output_dir:
            version = datetime.now().strftime("%Y%m%d_%H%M")
            roc_path = os.path.join(output_dir, f'roc_curve_v{version}.png')
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            plot_roc_curve(y_val, y_pred_proba, roc_path)
            mlflow.log_artifact(roc_path)
        
        # 모델 저장
        mlflow.sklearn.log_model(model, "random_forest_model")
    
    return model

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

def predict_and_save(model, test_data, output_path):
    """
    Make predictions and save them to a file
    
    Args:
        model: Trained model
        test_data (pandas.DataFrame): Test data
        output_path (str): Path to save predictions
    """
    print("\n🔄 예측 중...")
    
    # 특성 준비
    X_test = test_data.drop(['UID'], axis=1)
    categorical_cols = X_test.select_dtypes(include=['object']).columns
    X_test = pd.get_dummies(X_test, columns=categorical_cols)
    
    # 예측
    _, y_pred_proba = predict(model, X_test)
    
    # 결과 저장
    version = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = os.path.dirname(output_path)
    output_name = os.path.splitext(os.path.basename(output_path))[0]
    versioned_path = os.path.join(output_dir, f'{output_name}_v{version}.csv')
    
    submission = pd.DataFrame({
        'UID': test_data['UID'],
        '채무 불이행 확률': y_pred_proba
    })
    submission.to_csv(versioned_path, index=False)
    
    # MLflow에 예측 결과 저장
    with mlflow.start_run(nested=True):
        mlflow.log_artifact(versioned_path)
    
    print(f"\n✅ 예측 결과가 {versioned_path}에 저장되었습니다.") 