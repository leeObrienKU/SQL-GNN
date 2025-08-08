import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

def train_and_evaluate(model, data, train_loader, epochs, lr, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    best_val_acc = 0
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.to(device))
            pred = out.argmax(dim=1)
            val_acc = accuracy_score(
                data.y[data.val_mask].cpu().numpy(),
                pred[data.val_mask].cpu().numpy()
            )
        
        # Logging
        avg_loss = total_loss / len(train_loader)
        logger.log_metrics(epoch+1, avg_loss, val_acc, optimizer.param_groups[0]['lr'])
        
    # Final evaluation
    test_acc = accuracy_score(
        data.y[data.test_mask].cpu().numpy(),
        model(data.to(device))[data.test_mask].argmax(dim=1).cpu().numpy()
    )
    return test_acc