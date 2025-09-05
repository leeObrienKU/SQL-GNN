# utils/plots.py
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_training_curves(training_log, logger):
    """Plot training curves with consistent naming
    
    Args:
        training_log: list of dicts with keys: epoch, train_loss, val_acc
        logger: ExperimentLogger instance for consistent naming and directory structure
    """
    if not training_log:
        return
    
    epochs = [e["epoch"] for e in training_log]
    train_loss = [e["train_loss"] for e in training_log]
    val_acc = [e["val_acc"] for e in training_log]
    
    # Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train Loss", color='#2ecc71')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss - {logger.experiment_id}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = logger.save_plot(plt.gcf(), "training_loss")
    plt.close()
    
    # Val accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_acc, label="Val Accuracy", color='#3498db')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Validation Accuracy - {logger.experiment_id}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = logger.save_plot(plt.gcf(), "val_accuracy")
    plt.close()
    
    return loss_path, acc_path

def plot_confusion_matrix(cm, class_names, logger, normalize=False):
    """Plot confusion matrix with consistent naming
    
    Args:
        cm: numpy array of confusion matrix
        class_names: list of class names
        logger: ExperimentLogger instance for consistent naming and directory structure
        normalize: whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=class_names, yticklabels=class_names,
                cmap='YlOrRd')
    plt.title(f"Confusion Matrix - {logger.experiment_id}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save both normalized and raw versions if normalizing
    paths = []
    if normalize:
        paths.append(logger.save_plot(plt.gcf(), "confusion_matrix_norm"))
        plt.close()
        
        # Plot raw counts
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=class_names, yticklabels=class_names,
                    cmap='YlOrRd')
        plt.title(f"Confusion Matrix (Counts) - {logger.experiment_id}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        paths.append(logger.save_plot(plt.gcf(), "confusion_matrix_raw"))
    else:
        paths.append(logger.save_plot(plt.gcf(), "confusion_matrix"))
    
    plt.close()
    return paths

def plot_roc_curve(fpr, tpr, roc_auc, logger):
    """Plot ROC curve with consistent naming"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#e74c3c', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {logger.experiment_id}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path = logger.save_plot(plt.gcf(), "roc_curve")
    plt.close()
    return path

def plot_pr_curve(y_true, y_scores, logger):
    """Plot Precision-Recall curve with consistent naming"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#9b59b6', lw=2,
             label=f'PR curve (AP = {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {logger.experiment_id}')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path = logger.save_plot(plt.gcf(), "pr_curve")
    plt.close()
    return path, ap