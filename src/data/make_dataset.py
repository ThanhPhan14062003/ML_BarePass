import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

# Label mapping dictionary
label_dict = {
    'brain_glioma': 0,
    'brain_menin': 1,
    'brain_tumor': 2
}

# Load pre-trained models (without fully connected layers)
efficientnet_b0 = models.efficientnet_b0(pretrained=True)
efficientnet_b0 = nn.Sequential(*list(efficientnet_b0.children())[:-1])  # Remove classifier

vgg19 = models.vgg19(pretrained=True)
vgg19 = nn.Sequential(*list(vgg19.children())[:-1])  # Remove classifier

# Set models to evaluation mode
efficientnet_b0.eval()
vgg19.eval()

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
efficientnet_b0.to(device)
vgg19.to(device)

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

    b0_features = []
    vgg_features = []
    labels = []

    for img_path, label in tqdm(zip(df["filepath"], df["labels"])):
        try:
            # Load image
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

            with torch.no_grad():
                # Extract EfficientNet-B0 features
                b0_feat = efficientnet_b0(image)
                b0_feat = torch.flatten(b0_feat, start_dim=1)  # Flatten feature map

                # Extract VGG19 features
                vgg_feat = vgg19(image)
                vgg_feat = torch.flatten(vgg_feat, start_dim=1)  # Flatten feature map

            # Store features
            b0_features.append(b0_feat.cpu().numpy())
            vgg_features.append(vgg_feat.cpu().numpy())

            # Store label
            labels.append(label_dict[label])

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

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
