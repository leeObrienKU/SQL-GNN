
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

def train_and_evaluate(model, data, train_loader, epochs, lr, logger, clip_grad=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            try:
                batch = batch.to(device)
                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    out = model(batch)
                    loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])

                if scaler:
                    scaler.scale(loss).backward()
                    if clip_grad:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_grad:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    optimizer.step()

                total_loss += loss.item()
                del batch, out, loss
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("OOM error, skipping batch.")
                    torch.cuda.empty_cache()
                else:
                    raise e

        model.eval()
        with torch.no_grad():
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            val_acc = accuracy_score(
                data.y[data.val_mask].cpu().numpy(),
                pred[data.val_mask].cpu().numpy()
            )
            del out, pred
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        logger.log_metrics(epoch+1, avg_loss, val_acc, optimizer.param_groups[0]['lr'])

    with torch.no_grad():
        out = model(data.to(device))
        test_acc = accuracy_score(
            data.y[data.test_mask].cpu().numpy(),
            out[data.test_mask].argmax(dim=1).cpu().numpy()
        )
        del out
        torch.cuda.empty_cache()

    return test_acc
