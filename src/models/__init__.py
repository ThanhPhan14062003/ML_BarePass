import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import torch.nn.functional as F

def load_model(model_name, input_dim, num_classes):
    """
    Load and initialize a model based on the given model name.

    :param model_name: The name of the model to load.
    :param input_dim: The input feature dimension.
    :param num_classes: The number of output classes.
    :return: An instance of the specified model.
    """
    model_mapping = {
        "mlp": SimpleMLP,
        "decision_tree": DecisionTreeModel,
        "naive_bayes": NaiveBayesModel,
        "hmm": HMMModel,
        "bayesian_network": BayesianNetworkModel,
    }

    if model_name not in model_mapping:
        raise ValueError(
            f"Invalid model name '{model_name}'. Choose from {list(model_mapping.keys())}."
        )

    return model_mapping[model_name](input_dim, num_classes)


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
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class NaiveBayesModel:
    def __init__(self, input_dim=None, num_classes=None):
        self.model = GaussianNB()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
class HMMModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(HMMModel, self).__init__()
        self.num_classes = num_classes  # Number of hidden states
        self.input_dim = input_dim  # Feature dimension
        
        # Emission network (MLP)
        self.emission = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # Transition probabilities (learnable)
        self.transition_matrix = nn.Parameter(torch.randn(num_classes, num_classes))

        # Initial state probabilities (learnable)
        self.start_probs = nn.Parameter(torch.randn(num_classes))

    def forward(self, x):
        emissions = self.emission(x)  # Emission probabilities
        transition_probs = torch.softmax(self.transition_matrix, dim=-1)  # Normalize transition matrix
        start_probs = torch.softmax(self.start_probs, dim=0)  # Normalize initial state probabilities
        return emissions, transition_probs, start_probs
    
    
class BayesianNetworkModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BayesianNetworkModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Bayesian layers (mean and variance for weights)
        self.fc_mean = nn.Linear(input_dim, num_classes)
        self.fc_logvar = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # Sample weights from Gaussian distribution
        weight_mean = self.fc_mean(x)
        weight_logvar = self.fc_logvar(x)  # Log variance for numerical stability
        weight_std = torch.exp(0.5 * weight_logvar)  # Convert log variance to std
        
        # Reparameterization trick: sample from N(0,1) and scale
        epsilon = torch.randn_like(weight_std)
        sampled_output = weight_mean + weight_std * epsilon  # Bayesian output

        return F.log_softmax(sampled_output, dim=1)  # Apply softmax for classification
