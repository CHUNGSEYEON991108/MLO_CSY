import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def calculate_roc_auc(y_true, y_pred_proba):
    """
    Calculate ROC-AUC score
    
    Args:
        y_true (array-like): True binary labels
        y_pred_proba (array-like): Predicted probabilities
        
    Returns:
        float: ROC-AUC score
    """
    return roc_auc_score(y_true, y_pred_proba)

def plot_roc_curve(y_true, y_pred_proba, save_path):
    """
    Plot ROC curve and save it
    
    Args:
        y_true (array-like): True binary labels
        y_pred_proba (array-like): Predicted probabilities
        save_path (str): Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = calculate_roc_auc(y_true, y_pred_proba)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close() 