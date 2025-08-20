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
