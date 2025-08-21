# utils/plots.py
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(training_log, out_dir):
    """
    training_log: list of dicts with keys: epoch, train_loss, val_acc
    """
    if not training_log:
        return
    os.makedirs(out_dir, exist_ok=True)

    epochs = [e["epoch"] for e in training_log]
    train_loss = [e["train_loss"] for e in training_log]
    val_acc = [e["val_acc"] for e in training_log]

    # Loss curve
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_loss.png"))
    plt.close()

    # Val accuracy curve
    plt.figure()
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_accuracy.png"))
    plt.close()


def plot_confusion_matrix(cm, class_names, out_path, normalize=False):
    """
    cm: 2D numpy array
    class_names: list of class names (e.g., ["Stay", "Leave"])
    normalize: if True, show row-normalized percents
    """
    import itertools

    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size else 0.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = cm[i, j]
        plt.text(j, i, format(value, fmt),
                 horizontalalignment="center",
                 color="white" if value > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc_curve(y_true, y_scores, save_path):
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Attrition)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_pr_curve(y_true, y_scores, save_path):
    from sklearn.metrics import average_precision_score, precision_recall_curve
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Attrition)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
