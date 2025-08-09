# models/trainer.py
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix

@torch.no_grad()
def _evaluate(model, data, mask):
    model.eval()
    out = model(data)                  # logits [N, C]
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


def train_and_evaluate(model, data, train_loader, epochs, lr, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)

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

            loss = F.nll_loss(out[train_sel], y[train_sel])
            loss.backward()
            optimizer.step()

            bs = int(train_sel.sum().item())
            epoch_loss += float(loss.item()) * bs
            total_examples += bs

        epoch_loss = epoch_loss / max(1, total_examples)

        # Validation on the full graph
        val_acc, _, _, _, _ = _evaluate(model, data, data.val_mask)

        # Log
        logger.log_metrics(epoch, epoch_loss, val_acc, optimizer.param_groups[0]["lr"])

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

    # Final test metrics
    test_acc, test_f1, cm, y_true_m, y_pred_m = _evaluate(model, data, data.test_mask)

    # Stash in logger.metrics for plotting/saving in main
    logger.metrics["test_f1_macro"] = float(test_f1)
    logger.metrics["confusion_matrix"] = cm.astype(int).tolist()

    print(f"🧪  Test Accuracy: {test_acc:.4f}")
    print(f"🧪  Test F1 (macro): {test_f1:.4f}")

    return test_acc
