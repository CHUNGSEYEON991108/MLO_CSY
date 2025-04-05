import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

def calculate_roc_auc(y_true, y_pred_proba):
    """ROC-AUC 점수를 계산합니다."""
    return roc_auc_score(y_true, y_pred_proba)

def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """ROC 곡선을 그리고 선택적으로 저장합니다."""
    # ROC 곡선 계산
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def print_model_performance(y_true, y_pred_proba):
    """모델 성능을 출력합니다."""
    roc_auc = calculate_roc_auc(y_true, y_pred_proba)
    print(f"\n📊 모델 성능:")
    print(f"ROC-AUC 점수: {roc_auc:.4f}")
    return roc_auc 