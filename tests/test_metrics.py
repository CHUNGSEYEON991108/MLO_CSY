import pytest
import numpy as np
from mlops_csy.metrics import calculate_roc_auc, plot_roc_curve

def test_calculate_roc_auc():
    """ROC-AUC 계산 테스트"""
    # 테스트 데이터 생성
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    
    # ROC-AUC 계산
    roc_auc = calculate_roc_auc(y_true, y_pred_proba)
    
    # 결과 검증
    assert isinstance(roc_auc, float)
    assert 0 <= roc_auc <= 1
    assert roc_auc > 0.5  # 좋은 예측의 경우

def test_plot_roc_curve():
    """ROC 곡선 그리기 테스트"""
    # 테스트 데이터 생성
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    
    # 임시 파일 경로
    save_path = 'test_roc_curve.png'
    
    try:
        # ROC 곡선 그리기 및 저장
        plot_roc_curve(y_true, y_pred_proba, save_path)
        
        # 파일이 생성되었는지 확인
        import os
        assert os.path.exists(save_path)
    finally:
        # 테스트 후 임시 파일 삭제
        if os.path.exists(save_path):
            os.remove(save_path) 