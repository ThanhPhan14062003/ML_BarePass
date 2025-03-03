import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV
import inspect
import joblib
import sys
import os
# Abstract base class
class SKLearnModel():
    def __init__(self, model_class, **kwargs):
        self.model_class = model_class
        # Check if valid parameters are passed to the model
        valid_params = self._get_valid_params()
        self._validate_kwargs(kwargs, valid_params)
        # Initialize the model with valid kwargs
        self.model = model_class(**kwargs)
        print(self.model)

    def _get_valid_params(self):
        """Get valid hyperparameters for the model."""
        model_signature = inspect.signature(self.model_class)
        valid_params = model_signature.parameters.keys()
        return valid_params

    def _validate_kwargs(self, kwargs, valid_params):
        """Check that the keys in kwargs are valid."""
        invalid_params = [param for param in kwargs if param not in valid_params]
        if invalid_params:
            raise ValueError(f"Invalid parameters: [{', '.join(invalid_params)}] for {self.model_class.__name__}. "
                             f"Valid parameters are: [\n\t{'\n\t'.join(valid_params)}\n]")
    
    def fit(self, X_train, y_train):
        """Common fit method for all SKLearn models."""
        # Convert to NumPy if X_train is a tensor
        X_train_np = X_train.cpu().numpy() if torch.is_tensor(X_train) else X_train
        y_train_np = y_train.cpu().numpy() if torch.is_tensor(y_train) else y_train
        self.model.fit(X_train_np, y_train_np)

    def hyper_fit(self, X_train, y_train, param_grid = {}):
        X_train_np = X_train.cpu().numpy() if torch.is_tensor(X_train) else X_train
        y_train_np = y_train.cpu().numpy() if torch.is_tensor(y_train) else y_train
        
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train_np, y_train_np)
        self.model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")
        return self
    
    def predict(self, X):
        """Common predict method for all SKLearn models."""
        # Convert to NumPy if X is a tensor
        X_np = X.cpu().numpy() if torch.is_tensor(X) else X
        preds = self.model.predict(X_np)
        return torch.tensor(preds, device=X.device)  # Return predictions as a tensor

    def predict_proba(self, X):
        """Common predict_proba method for all SKLearn models."""
        # Convert to NumPy if X is a tensor
        X_np = X.cpu().numpy() if torch.is_tensor(X) else X
        probs = self.model.predict_proba(X_np)
        return torch.tensor(probs, device=X.device)  # Return probabilities as a tensor
    
    def save_model(self,dir):
        joblib.dump(self.model, dir)

    def __get_name__(self):
        return self.model_class.__name__
    
__skl_model__ = {
    'decision_tree': DecisionTreeClassifier,
    "naive_bayes": GaussianNB,
    "svm": SVC()
}

class TorchModel():
    def _check_valid_param(self,**kwargs):
        valid_keys = ['input_dim', 'num_classes']
        
        # Find invalid keys in kwargs
        invalid_keys = [key for key in kwargs if key not in valid_keys]
        
        # If there are invalid keys, raise an error with a message
        if invalid_keys:
            raise ValueError(f"Invalid parameters: {', '.join(invalid_keys)}. Valid parameters are: {', '.join(valid_keys)}.")        

        assert 'input_dim' in kwargs, "'input_dim' must be provided."
        assert 'num_classes' in kwargs, "'num_classes' must be provided."
        
        assert isinstance(kwargs['input_dim'], int) and kwargs['input_dim'] > 0, "'input_dim' must be a positive integer."
        assert isinstance(kwargs['num_classes'], int) and kwargs['num_classes'] > 0, "'num_classes' must be a positive integer."
        
class SimpleMLP(TorchModel, nn.Module):
    def __init__(self, **kwargs):
        self._check_valid_param(**kwargs)

        super(SimpleMLP, self).__init__()
        
        input_dim = kwargs['input_dim']
        num_classes = kwargs['num_classes']
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)
    
class HMMModel(TorchModel, nn.Module):
    def __init__(self, **kwargs):
        self._check_valid_param(**kwargs)
        
        super(HMMModel, self).__init__()
        self.num_classes = kwargs['num_classes']  # Number of hidden states
        self.input_dim = kwargs['input_dim']  # Feature dimension
        
        # Emission network (MLP)
        self.emission = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )
        
        # Transition probabilities (learnable)
        self.transition_matrix = nn.Parameter(torch.randn(self.num_classes, self.num_classes))

        # Initial state probabilities (learnable)
        self.start_probs = nn.Parameter(torch.randn(self.num_classes))

    def forward(self, x):
        emissions = self.emission(x)  # Emission probabilities
        transition_probs = torch.softmax(self.transition_matrix, dim=-1)  # Normalize transition matrix
        start_probs = torch.softmax(self.start_probs, dim=0)  # Normalize initial state probabilities
        return emissions, transition_probs, start_probs
    
    
class BayesianNetworkModel(TorchModel, nn.Module):
    def __init__(self, **kwargs):
        self._check_valid_param(**kwargs)
        
        super(BayesianNetworkModel, self).__init__()
        self.num_classes = kwargs['num_classes']  # Number of hidden states
        self.input_dim = kwargs['input_dim']  # Feature dimension

        # Bayesian layers (mean and variance for weights)
        self.fc_mean = nn.Linear(self.input_dim, self.num_classes)
        self.fc_logvar = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, x):
        # Sample weights from Gaussian distribution
        weight_mean = self.fc_mean(x)
        weight_logvar = self.fc_logvar(x)  # Log variance for numerical stability
        weight_std = torch.exp(0.5 * weight_logvar)  # Convert log variance to std
        
        # Reparameterization trick: sample from N(0,1) and scale
        epsilon = torch.randn_like(weight_std)
        sampled_output = weight_mean + weight_std * epsilon  # Bayesian output

        return F.log_softmax(sampled_output, dim=1)  # Apply softmax for classification

__other_model__ = {
    'mlp': SimpleMLP,
    "hmm": HMMModel,
    "bayesian_network": BayesianNetworkModel,
}
    
def load_model(model_name, **kwargs):
    """
    Load and initialize a model based on the given model name.

    :param model_name: The name of the model to load.
    :param kwargs: The hyper parameters dictionary of the model.
    :return: An instance of the specified model.
    """
    valid_names = list(__skl_model__.keys()) + list(__other_model__.keys())
    
    # Check if model_name is valid
    assert model_name in valid_names, f"Invalid model_name '{model_name}'. Valid options are: {', '.join(valid_names)}"
    
    try:
        # Handle other models (e.g., MLP)
        if model_name in list(__other_model__.keys()):
            return __other_model__[model_name](**kwargs)
        
        # Handle SKLearn models
        return SKLearnModel(__skl_model__[model_name],**kwargs)
    except Exception as e:
        print(e)
        sys.exit(1)
        
        