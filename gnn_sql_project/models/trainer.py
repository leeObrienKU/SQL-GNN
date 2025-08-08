import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix

def train_and_evaluate(model, data, train_loader, epochs, lr, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    # --- Weighted loss for class imbalance ---
    num_emp = int(getattr(data, 'num_employees', data.num_nodes))
    y_emp = data.y[:num_emp].to(torch.long)
    binc = torch.bincount(y_emp)
    weights = (1.0 / (binc.float().clamp(min=1.0)))
    weights = weights / weights.mean()
    weights = weights.to(device)
    criterion = nn.NLLLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Early stopping setup ---
    best_val = -1.0
    best_state = None
    patience, bad = 20, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation accuracy
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1)
            val_correct = int((pred[data.val_mask] == data.y[data.val_mask]).sum())
            val_total = int(data.val_mask.sum())
            val_acc = val_correct / max(val_total, 1)

        logger.log_metrics(epoch, total_loss / len(train_loader), val_acc, optimizer.param_groups[0]['lr'])

        # Early stopping check
        if val_acc > best_val + 1e-4:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            print(f"⛔ Early stop at epoch {epoch} (best val={best_val:.4f})")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test evaluation
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1).cpu().numpy()
        true = data.y.cpu().numpy()

    num_emp = int(getattr(data, 'num_employees', data.num_nodes))
    pred_emp = pred[:num_emp]
    true_emp = true[:num_emp]
    test_acc = (pred_emp == true_emp).mean()
    test_f1_macro = f1_score(true_emp, pred_emp, average='macro')
    cm = confusion_matrix(true_emp, pred_emp)

    print(f"🧪 Test Accuracy: {test_acc:.4f}")
    print(f"🧪 Test F1 (macro): {test_f1_macro:.4f}")

    logger.metrics["test_f1_macro"] = float(test_f1_macro)
    logger.metrics["confusion_matrix"] = cm.astype(int).tolist()

    return test_acc
