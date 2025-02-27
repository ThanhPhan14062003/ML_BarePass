import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier

def load_model(model_name, input_dim, num_classes):
    if model_name == "mlp":
        return SimpleMLP(input_dim, num_classes)
    elif model_name == "decision_tree":
        return DecisionTreeModel(input_dim, num_classes)
    else:
        raise ValueError("Invalid model name. Choose 'mlp' or 'decision_tree'.")

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

class DecisionTreeModel:
    def __init__(self, input_dim=None, num_classes=None):
        self.model = DecisionTreeClassifier(random_state=42)

    def fit(self, X_train, y_train):
        X_train_np = X_train.cpu().numpy() if torch.is_tensor(X_train) else X_train
        y_train_np = y_train.cpu().numpy() if torch.is_tensor(y_train) else y_train
        self.model.fit(X_train_np, y_train_np)

    def predict(self, X):
        X_np = X.cpu().numpy() if torch.is_tensor(X) else X
        preds = self.model.predict(X_np)
        return torch.tensor(preds, device=X.device)  # Keep tensor format

    def predict_proba(self, X):
        X_np = X.cpu().numpy() if torch.is_tensor(X) else X
        probs = self.model.predict_proba(X_np)
        return torch.tensor(probs, device=X.device)  # Keep tensor format