import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

def ensure_results_directories(base_directory):
    subdirs = ['loss', 'accuracy', 'confusion_matrix', 'roc_auc_score', 'classification_report']
    for subdir in subdirs:
        dir_path = os.path.join(base_directory, subdir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def plot_confusion_matrix(y_true, y_pred, classes, architecture_name, base_directory):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    filepath = os.path.join(base_directory, 'confusion_matrix', f'{architecture_name}_confusion_matrix.png')
    plt.savefig(filepath)
    plt.close()

def plot_metrics(history, architecture_name, base_directory):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['test_loss'], 'r-', label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.tight_layout()
    filepath_loss = os.path.join(base_directory, 'loss', f'{architecture_name}_loss_curve.png')
    plt.savefig(filepath_loss)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['test_accuracy'], 'r-', label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.tight_layout()
    filepath_acc = os.path.join(base_directory, 'accuracy', f'{architecture_name}_accuracy_curve.png')
    plt.savefig(filepath_acc)
    plt.close()

def evaluate_model(y_true, y_pred, y_prob, classes, history, architecture_name, base_directory='../result'):
    ensure_results_directories(base_directory)

    plot_confusion_matrix(y_true, y_pred, classes, architecture_name, base_directory)

    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    with open(os.path.join(base_directory, 'classification_report', f'{architecture_name}_classification_report.txt'), 'w') as file:
        file.write(report)

    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        with open(os.path.join(base_directory, 'roc_auc_score', f'{architecture_name}_roc_auc_score.txt'), 'w') as file:
            file.write(f"ROC-AUC Score: {roc_auc:.4f}")
    except Exception as e:
        with open(os.path.join(base_directory, 'roc_auc_score', f'{architecture_name}_roc_auc_score.txt'), 'w') as file:
            file.write(f"ROC-AUC could not be computed: {e}")

    accuracy = accuracy_score(y_true, y_pred)
    with open(os.path.join(base_directory, 'accuracy', f'{architecture_name}_accuracy_score.txt'), 'w') as file:
        file.write(f"Accuracy Score: {accuracy:.4f}")

    plot_metrics(history, architecture_name, base_directory)

    print(f"\nModel Performance Summary ({architecture_name}):")
    print(f"Highest Train Accuracy: {max(history['train_accuracy'])*100:.2f}%")
    print(f"Highest Test Accuracy: {max(history['test_accuracy'])*100:.2f}%")
    print(f"Lowest Train Loss: {min(history['train_loss']):.4f}")
    print(f"Lowest Test Loss: {min(history['test_loss']):.4f}")
