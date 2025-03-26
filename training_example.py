import torch
from torch.optim import Adam
from loss import GaussianNLLLoss
from master import MASTERModel

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    mse_loss = 0.0

    with torch.no_grad():
        for data, target in val_loader:
            mean_pred, std = model(data)  # During inference, returns mean and std
            log_var = torch.log(std**2 + 1e-6)  # Convert std back to log_var, add epsilon
            val_loss += criterion(mean_pred, log_var, target).item()
            mse_loss += torch.mean((mean_pred - target)**2).item()

    return val_loss / len(val_loader), mse_loss / len(val_loader)

# Example training loop
def train(model, train_loader, val_loader, epochs=10, lr=0.001):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = GaussianNLLLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # Model returns mean prediction and log variance
            mean_pred, log_var = model(data)

            # Calculate loss
            loss = criterion(mean_pred, log_var, target)

            # Backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        val_loss, mse_loss = validate(model, val_loader, criterion)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss:.4f}, MSE: {mse_loss:.4f}')
