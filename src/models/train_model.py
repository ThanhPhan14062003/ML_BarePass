import torch
import torch.nn as nn
import torch.optim as optim
from src.models import SimpleMLP, HMMModel, BayesianNetworkModel, SKLearnModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
import os
import joblib
import torch.optim as optim
import torch.nn.functional as F

def train_sklearn(model: SKLearnModel, train_loader, hyper_tune=False, param_grid={}):
    os.makedirs("weights", exist_ok=True)
    X_train, y_train = next(iter(train_loader))
    
    if hyper_tune:
        model = model.hyper_fit(X_train,y_train,param_grid)
    else:
        model.fit(X_train, y_train)
    
    print(f"{model.__get_name__()} training complete.")
    
    y_pred_proba = model.predict_proba(X_train)
    loss = log_loss(y_train.cpu().numpy(), y_pred_proba.cpu().numpy())
    y_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train.cpu().numpy(), y_pred.cpu().numpy())
    
    print(f"Log Loss: {loss:.4f}, Training Accuracy: {train_acc:.4f}")
    model.save_model(os.path.join("weights", f"{model.__get_name__()}.pkl"))

def train_mlp(model: SimpleMLP, train_loader, criterion, optimizer, device, epochs):
    model.to(device)
    model.train()
    os.makedirs("weights", exist_ok=True)
    
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}, Accuracy: {correct/total:.4f}")
    
    torch.save(model.state_dict(), os.path.join("weights", f"{model.__class__.__name__}.pth"))
    
def train_hmm(model: HMMModel, train_loader, device="cpu", epochs=50):
    model.to(device)
    model.train()
    
    # Use NLLLoss (Negative Log-Likelihood)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            # Get log probabilities from the model
            log_probs, _, _ = model(X_batch)

            # Ensure correct shape for NLLLoss
            y_batch = y_batch.view(-1)

            # Compute loss
            loss = criterion(log_probs, y_batch)

            # Convert to a more readable scale (optional)
            readable_loss = torch.exp(loss)  # Converts log-space loss to normal space

            loss.backward()
            optimizer.step()
            
            total_loss += readable_loss.item()

        # Print more understandable loss
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Adjusted Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), f"weights/{model.__class__.__name__}.pth")

def train_bayesian_network(model: BayesianNetworkModel, train_loader, device="cpu", epochs=50):
    model.to(device)
    model.train()

    # Negative Log Likelihood Loss for classification
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            
            log_probs = model(X_batch)  # Get log probabilities

            loss = criterion(log_probs, y_batch)  # Compute loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), f"weights/{model.__class__.__name__}.pth")
    print("Training complete. Model saved.")


def train_model(model, train_loader, criterion=None, optimizer=None, device="cpu", epochs=100, hyper_tune=False, param_grid={}):
    """
    Train the given model based on its type.

    :param model: The model to be trained.
    :param train_loader: DataLoader providing training data.
    :param criterion: Loss function (only used for models requiring it). Ignore for sklearn models.
    :param optimizer: Optimizer (only used for models requiring it). Ignore for sklearn models.
    :param device: Device to run training on ("cpu" or "cuda").
    :param epochs: Number of training epochs. Ignore for sklearn models.
    :param hyper_tune: whether to apply hyper parameters tuning for sklearn models. Ignore if model is not sklearn model.
    :param param_grid: param_grid parameter for hyper parameters tuning (use by GridSearchCV)
    """
    if isinstance(model, SimpleMLP):
        train_mlp(model, train_loader, criterion, optimizer, device, epochs)
    elif isinstance(model, SKLearnModel):
        train_sklearn(model, train_loader,hyper_tune,param_grid)
    elif isinstance(model, HMMModel):
        train_hmm(model, train_loader, device=device, epochs=epochs)
    elif isinstance(model, BayesianNetworkModel):
        train_bayesian_network(model, train_loader, epochs=epochs)
    else:
        raise ValueError(f"Unsupported model type: {type(model).__name__}")