import numpy as np
import pandas as pd

from keras.api.applications import EfficientNetB0
from keras.api.applications.vgg19 import VGG19
from keras.api.models import Model
from keras.api.layers import Dense, GlobalAveragePooling2D


# Create EfficientNetB0 model for feature extraction
def create_efficientnet_model():
    # Load EfficientNetB0 model with weights pre-trained on ImageNet
    efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D()(efficientnet_model.output)
    # Extract features from the 'global_average_pooling2d' layer, which is before the final dense layer 
    return Model(inputs=efficientnet_model.input, outputs=x)

# Create VGG-19 model for feature extraction
def create_vgg_model():
    vgg_model = VGG19(weights='imagenet', include_top= False)
    x = GlobalAveragePooling2D()(vgg_model.output)
    return Model(inputs=vgg_model.input, outputs=x)

# Add a new fully connected layer with 2 neurons for binary classification
def add_custom_fc_layer(model):
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='relu')(x)  # 1000 neurons before final layer (as requested)
    x = Dense(2, activation='softmax')(x)  # 2 neurons for binary classification
    return Model(inputs=model.input, outputs=x)

# Function to extract features from both EfficientNetB0 and VGG-19 and fuse them
def extract_features(df: pd.DataFrame, preprocess_images):
    # Load models
    effnet_model = create_efficientnet_model()
    vgg_model = create_vgg_model()
    
    # Preprocess images for both models
    image_paths = df['filepath'].values
    
    effnet_images = preprocess_images(image_paths, model_type='effnet')
    
    vgg_images = preprocess_images(image_paths, model_type='vgg')
    
    # Extract features from both models
    effnet_features = effnet_model.predict(effnet_images)
    vgg_features = vgg_model.predict(vgg_images)
    
    return effnet_features, vgg_features
