import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from src.data.make_dataset import label_dict
import torch
import numpy as np
from src.models import *


def evaluate_sklearn_model(model, X_test, y_test, visualize=False):
    """
    Evaluate the performance of a trained SVC model.

    Parameters:
    - model: Trained sklearn model.
    - X_test: Test features.
    - y_test: True labels for the test set.
    - label_dict: Dictionary mapping label names to indices.
    - visualize: If True, displays the confusion matrix.

    Returns:
    - None
    """
    # Reverse the label_dict to map indices back to label names
    inv_label_dict = {v: k for k, v in label_dict.items()}

    # Predict the labels for the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[inv_label_dict[i] for i in range(len(label_dict))]))

    # Print the individual metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Visualize the confusion matrix if requested
    if visualize:
        cm = confusion_matrix(y_test, y_pred, labels=list(label_dict.values()))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_dict.keys(), yticklabels=label_dict.keys())
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1

    }
        
        
def evaluate_torch_model(model, test_loader, device="cpu", visualization=True):
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass to get model predictions
            outputs = model(X_batch)
            
            # Get predictions: the output will differ based on model
            if isinstance(model, (SimpleMLP, BayesianNetworkModel)):
                _, predicted = torch.max(outputs, 1)  # For MLP and Bayesian, use argmax for classification
            elif isinstance(model, HMMModel):
                log_probs, _, _ = outputs
                _, predicted = torch.max(log_probs, 1)  # HMM outputs log-probabilities
            
            # Store predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')

    # Print classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))

    # Print accuracy, f1, recall, and precision scores
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")

    # If visualization is enabled, show the confusion matrix
    if visualization:
        cm = confusion_matrix(all_labels, all_preds)
        # Map the labels to the strings using label_dict
        labels = list(label_dict.keys())
        
        # Plot confusion matrix with mapped labels
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1

    }
