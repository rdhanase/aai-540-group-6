import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import joblib
import os

def evaluate_model(model, X_test, y_test, output_dir='aai-540-group-6/models/evaluation'):
    """
    Generates metrics and visualizations for the trained model.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 1. Classification Report
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
        
    # 2. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
    plt.close()
    
    print(f"Evaluation artifacts saved to {output_dir}")

if __name__ == "__main__":
    # Local validation if model exists
    model_path = 'aai-540-group-6/models/rf_baseline.joblib'
    if os.path.exists(model_path):
        # We'd need X_test and y_test, which are returned by train.py
        # For a standalone test, we could reload a sample of data
        print("Run train.py to generate model and test data.")
    else:
        print("Model not found. Run train.py first.")
