import pytest
import mlflow
import os
from mlops_csy.experiment_tracker import ExperimentTracker, get_best_model

def test_experiment_tracker_initialization():
    """실험 트래커 초기화 테스트"""
    tracker = ExperimentTracker("test_experiment")
    assert tracker.experiment_name == "test_experiment"

def test_experiment_tracker_logging():
    """실험 트래커 로깅 기능 테스트"""
    tracker = ExperimentTracker("test_experiment")
    
    # 테스트용 데이터
    model_params = {"n_estimators": 100, "max_depth": 10}
    metrics = {"roc_auc": 0.85}
    
    with tracker.start_run():
        # 파라미터 로깅
        tracker.log_model_params(model_params)
        
        # 메트릭 로깅
        tracker.log_metrics(metrics)
        
        # 임시 파일 생성 및 아티팩트 로깅
        test_file = "test_artifact.txt"
        with open(test_file, "w") as f:
            f.write("test artifact")
        
        try:
            tracker.log_artifact(test_file)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

def test_get_best_model():
    """최적 모델 가져오기 테스트"""
    # 존재하지 않는 실험에 대한 테스트
    assert get_best_model("non_existent_experiment") is None 