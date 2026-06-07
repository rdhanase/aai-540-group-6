import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import os

def evaluate_model(model, X_test, y_test, results_path='aai-540-group-6/models/evaluation'):
    """
    Generate metrics and plots for the model.
    Focus on Recall as per project design doc.
    """
    os.makedirs(results_path, exist_ok=True)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    # 1. Report
    report = classification_report(y_test, preds)
    print("Results summary:\n", report)
    with open(os.path.join(results_path, 'metrics.txt'), 'w') as f:
        f.write(report)
        
    # 2. CM Heatmap
    plt.figure(figsize=(7, 5))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Collision Confusion Matrix')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(results_path, 'cm.png'))
    plt.close()
    
    # 3. Precision-Recall
    prec, rec, _ = precision_recall_curve(y_test, probs)
    area = auc(rec, prec)
    
    plt.figure(figsize=(7, 5))
    plt.plot(rec, prec, label=f'AUC-PR = {area:.2f}')
    plt.title('PR Curve - Severity Classifier')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'pr_curve.png'))
    plt.close()
    
    print(f"Artifacts saved to {results_path}")

if __name__ == "__main__":
    print("Usage: Call from train.py flow.")
