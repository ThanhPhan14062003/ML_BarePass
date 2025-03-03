import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

# Create EfficientNetB0 model for feature extraction
def create_efficientnet_model(device= torch.device("cpu")):
    efficientnet_b0 = models.efficientnet_b0(pretrained=True)
    efficientnet_b0 = nn.Sequential(*list(efficientnet_b0.children())[:-2])  # Remove classifier

    return efficientnet_b0.eval().to(device)

# Create VGG-19 model for feature extraction
def create_vgg_model(device= torch.device("cpu")):
    vgg19 = models.vgg19(pretrained=True)
    vgg19 = nn.Sequential(*list(vgg19.children())[:-1])  # Remove classifier
    
    return vgg19.eval().to(device)

# Function to extract features from both EfficientNetB0 and VGG-19 and fuse them
def extract_features(model, image):
    
    with torch.no_grad():
        feature = model(image)
        feature = torch.flatten(feature, start_dim=1)
    
    return feature
