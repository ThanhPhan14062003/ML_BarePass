import numpy as np
import pandas as pd
from skimage import io, transform
from keras.api.applications import EfficientNetB0
from keras.api.applications.efficientnet import preprocess_input as preprocess_input_effnet
from keras.api.applications.vgg19 import VGG19
from keras.api.applications.vgg19 import preprocess_input as preprocess_input_vgg
from keras.api.models import Model
from keras.api.layers import Input
from src.features.build_features import extract_features
from src.data.preprocess import preprocess_images

label_dict = {
    'brain_glioma': 0,
    'brain_menin': 1,
    'brain_tumor': 2
}

def load_data(df: pd.DataFrame):
    """
    Load and preprocess image data for VGG-19 feature extraction.
    
    Args:
    - df (pd.DataFrame): A DataFrame with image paths and image labels.
    
    Returns:
    - dict: 
        b0_feature: EfficientNetB0 feature at layer MatMul (1000 neurons)
        vgg_feature: VGG19 feature at layer fc8 (1000 neurosn)
        fused_feature: concatenation of EfficientNetB0 feature and VGG19 feature
        labels: label of images
    """
    
    
    b0, vgg = extract_features(df,preprocess_images)
    fused_features = np.concatenate((b0, vgg), axis=1)
    
    # Get the labels from the DataFrame
    y = df['labels'].map(label_dict).values
    
    return {
        'b0_feature': b0,
        'vgg_feature': vgg,
        'fused_feature': fused_features,
        'labels': y
    }
        