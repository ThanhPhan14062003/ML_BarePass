import torch
import torch.nn as nn
import torch.optim as optim
from src.models import DecisionTreeModel, SimpleMLP
from sklearn.metrics import accuracy_score, log_loss
import os
import joblib

# ==========================
# Training Function (with Progress)
# ==========================
def train_model(model, train_loader, criterion=None, optimizer=None, device="cpu", epochs=100):
    os.makedirs("weights", exist_ok=True)
    if isinstance(model, SimpleMLP):  # If using MLP (Torch)
        model.to(device)
        model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

            train_acc = correct / total
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}, Accuracy: {train_acc:.4f}")
                
            torch.save(model.state_dict(), os.path.join("weights", "mlp.pth"))

    elif isinstance(model, DecisionTreeModel):  # If using Decision Tree (sklearn)
        X_train, y_train = next(iter(train_loader))  # Get entire dataset (one batch)
        model.fit(X_train, y_train)
        print("Decision Tree training complete.")

        # Compute log loss during training
        y_pred_proba = model.predict_proba(X_train)
        y_train_np = y_train.cpu().numpy()
        loss = log_loss(y_train_np, y_pred_proba.cpu().numpy())

        # Evaluate Decision Tree on Training Data
        y_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train_np, y_pred.cpu().numpy())
        
        print(f"Decision Tree Log Loss: {loss:.4f}, Training Accuracy: {train_acc:.4f}")
        
        joblib.dump(model, os.path.join("weights", "decision_tree.pkl"))

    else:
        raise ValueError("Unsupported model type.")