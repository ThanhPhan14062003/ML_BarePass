import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api.applications import EfficientNetB0
from keras.api.applications.vgg19 import VGG19
from keras.api.models import Model
from keras.api.layers import Dense, Flatten, BatchNormalization



# Create EfficientNetB0 model for feature extraction
def create_efficientnet_model():
    # Load EfficientNetB0 model with weights pre-trained on ImageNet
    efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False)
    x = Flatten()(efficientnet_model.output)
    
    model =  Model(inputs=efficientnet_model.input, outputs=x)
    return model

# Create VGG-19 model for feature extraction
def create_vgg_model():
    vgg_model = VGG19(weights='imagenet', include_top= False)
    x = Flatten()(vgg_model.output)
    
    model =  Model(inputs=vgg_model.input, outputs=x)
    return model
    
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
