import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from src.features.build_features import *
from src.data.preprocess import preprocess_images

# Label mapping dictionary
label_dict = {
    'brain_glioma': 0,
    'brain_menin': 1,
    'brain_tumor': 2
}

def load_data(df: pd.DataFrame):
    """
    Load and preprocess image data for VGG-19 and EfficientNet-B0 feature extraction.
    
    Args:
    - df (pd.DataFrame): A DataFrame with image paths and image labels.
    
    Returns:
    - dict: 
        b0_feature: EfficientNetB0 feature (flattened)
        vgg_feature: VGG19 feature (flattened)
        fused_feature: concatenation of EfficientNetB0 and VGG19 features
        labels: label of images
    """

    print("Loading images and extracting features...")

    device = get_device()
    print(f"Using device: {device}")
    vgg_19 = create_vgg_model(device)
    efficientnet_b0 = create_efficientnet_model(device)
    
    b0_features = []
    vgg_features = []
    labels = []

    for img_path, label in tqdm(zip(df["filepath"], df["labels"])):
        try:
            # Load image
            image = preprocess_images(img_path).unsqueeze(0).to(device)
           
            b0_feat = extract_features(efficientnet_b0, image)
            
            vgg_feat = extract_features(vgg_19, image)
            
            
            # Store features
            b0_features.append(b0_feat.cpu().numpy())
            vgg_features.append(vgg_feat.cpu().numpy())

            # Store label
            labels.append(label_dict[label])

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return

    # Convert lists to NumPy arrays
    b0_features = np.vstack(b0_features)
    vgg_features = np.vstack(vgg_features) 
    fused_features = np.concatenate((b0_features, vgg_features), axis=1)

    return {
        'b0_feature': b0_features,
        'vgg_feature': vgg_features,
        'fused_feature': fused_features,
        'labels': np.array(labels)
    }
