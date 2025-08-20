# models/trainer.py
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix

@torch.no_grad()
def _evaluate(model, data, mask, threshold: float | None = None):
    model.eval()
    out = model(data)                  # logits [N, C]
    # Default argmax; for attrition with binary classes, allow threshold on positive class prob
    if getattr(data, "task", None) == "attrition" and out.shape[1] == 2 and threshold is not None:
        p_pos = out.exp()[:, 1]
        y_pred = (p_pos >= float(threshold)).long()
    else:
        y_pred = out.argmax(dim=1)         # [N]
    y_true = data.y.view(-1)           # [N]
    mask = mask.bool()

    # Only score on masked nodes that have valid labels
    valid = mask & (y_true >= 0)
    if valid.sum() == 0:
        return 0.0, 0.0, np.array([[0, 0], [0, 0]]), y_true.cpu().numpy(), y_pred.cpu().numpy()

    y_true_m = y_true[valid].cpu().numpy()
    y_pred_m = y_pred[valid].cpu().numpy()

    acc = (y_true_m == y_pred_m).mean().item()
    # macro F1 supports multi-class; for binary it's the usual macro
    f1 = f1_score(y_true_m, y_pred_m, average="macro")
    # confusion matrix with natural class indexing
    labels = sorted(np.unique(y_true_m))
    cm = confusion_matrix(y_true_m, y_pred_m, labels=labels)

    return acc, f1, cm, y_true_m, y_pred_m


def train_and_evaluate(model, data, train_loader, epochs, lr, logger, pos_threshold: float = 0.5, auto_threshold: bool = False,
                       lr_decay_type: str = "exponential", lr_decay_gamma: float = 0.95, lr_decay_step_size: int = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    # Learning rate scheduler (optional)
    scheduler = None
    try:
        if lr_decay_type == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(lr_decay_gamma))
        elif lr_decay_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(lr_decay_step_size), gamma=float(lr_decay_gamma))
    except Exception:
        scheduler = None

    # Optional: class-weighted loss for imbalanced tasks (enabled for attrition)
    class_weight = None
    try:
        if getattr(data, "task", None) == "attrition":
            with torch.no_grad():
                y_all = data.y.view(-1)
                is_train = data.train_mask.bool()
                # restrict to employee nodes only
                num_employees = int(getattr(data, "num_employees", y_all.numel()))
                is_emp = torch.arange(y_all.numel(), device=device) < num_employees
                sel = is_train & (y_all >= 0) & is_emp
                if sel.sum() > 0:
                    y_train = y_all[sel].to(torch.long)
                    num_classes = int(getattr(data, "num_classes", 0))
                    if num_classes <= 0:
                        num_classes = int(y_train.max().item()) + 1
                    counts = torch.bincount(y_train, minlength=num_classes).float()
                    counts = torch.clamp(counts, min=1.0)
                    # Inverse frequency weights; higher weight for rarer classes
                    class_weight = (counts.sum() / counts).to(device)
                    # Normalize weights to have mean 1.0 for stability
                    class_weight = class_weight / class_weight.mean().clamp_min(1e-8)
    except Exception:
        # Fall back to unweighted loss if anything goes wrong computing weights
        class_weight = None

    best_val_acc = -1.0
    best_state = None
    patience = 20
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        total_examples = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)  # logits on the sampled subgraph
            # Only use nodes within the batch that are in global train_mask and have labels
            y = batch.y.view(-1)
            if hasattr(batch, "train_mask"):
                mask = batch.train_mask
            else:
                # fallback: everything in the batch is trainable
                mask = torch.ones_like(y, dtype=torch.bool)

            train_sel = mask & (y >= 0)

            if train_sel.sum() == 0:
                continue

            if class_weight is not None:
                loss = F.nll_loss(out[train_sel], y[train_sel], weight=class_weight)
            else:
                loss = F.nll_loss(out[train_sel], y[train_sel])
            loss.backward()
            optimizer.step()

            bs = int(train_sel.sum().item())
            epoch_loss += float(loss.item()) * bs
            total_examples += bs

        epoch_loss = epoch_loss / max(1, total_examples)

        # Validation on the full graph (use argmax during training for stability); also compute F1 for logging
        val_acc, val_f1, _, _, _ = _evaluate(model, data, data.val_mask, None)

        # Log
        logger.log_metrics(epoch, epoch_loss, val_acc, optimizer.param_groups[0]["lr"], val_f1=val_f1)

        # LR scheduler step (advance for next epoch)
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

        # Early stopping
        improved = val_acc > best_val_acc + 1e-6
        if improved:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"⛔  Early stop at epoch {epoch} (best val={best_val_acc:.4f})")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Select decision threshold for attrition if requested
    threshold_to_use = None
    if getattr(data, "task", None) == "attrition" and int(getattr(data, "num_classes", 2)) == 2:
        if auto_threshold:
            # sweep thresholds on validation set to maximize macro-F1
            candidates = torch.linspace(0.1, 0.9, steps=17).tolist()
            best_f1 = -1.0
            best_thr = pos_threshold
            for thr in candidates:
                _, f1_val, _, _, _ = _evaluate(model, data, data.val_mask, thr)
                if f1_val > best_f1 + 1e-9:
                    best_f1 = f1_val
                    best_thr = thr
            threshold_to_use = float(best_thr)
            logger.metrics["best_threshold_val_f1"] = float(best_f1)
            logger.metrics["best_threshold"] = float(threshold_to_use)
        else:
            threshold_to_use = float(pos_threshold)
            logger.metrics["best_threshold"] = float(threshold_to_use)

    # Final test metrics (apply threshold if available)
    test_acc, test_f1, cm, y_true_m, y_pred_m = _evaluate(model, data, data.test_mask, threshold_to_use)

    # Stash in logger.metrics for plotting/saving in main
    logger.metrics["test_f1_macro"] = float(test_f1)
    logger.metrics["confusion_matrix"] = cm.astype(int).tolist()

    print(f"🧪  Test Accuracy: {test_acc:.4f}")
    print(f"🧪  Test F1 (macro): {test_f1:.4f}")

    return test_acc
