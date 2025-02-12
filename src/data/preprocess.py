import cv2
import numpy as np
from skimage import io, transform, color
from keras.api.preprocessing.image import img_to_array
from keras.api.applications.vgg19 import preprocess_input as preprocess_input_vgg
from keras.api.applications.efficientnet import preprocess_input as preprocess_input_effnet

def crop_brain_contour(image):
    #Blurring using Gaussian
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Canny Edge Detection Algortihm from OpenCV
    thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        new_image = image[y:y+h, x:x+w]
        break 
    
    return new_image

# Preprocess images by cropping, resizing, and normalizing them
def preprocess_images(image_paths, model_type='effnet'):
    images = []
    for path in image_paths:
        image = io.imread(path)
        gray = color.rgb2gray(image)
        gray_image = (gray * 255).astype(np.uint8)
        cropped_image = crop_brain_contour(gray_image)
        image_resized = transform.resize(cropped_image, (224, 224), mode='reflect', anti_aliasing=True)
        image_array = img_to_array(image_resized)
        image_array = np.expand_dims(image_array, axis=0)
        
        # Convert grayscale to RGB if the image is grayscale, for VGG-19 model only
        if len(image_array.shape) == 4 and image_array.shape[-1] == 1 and model_type == 'vgg':  # Check if grayscale
            image_array = np.repeat(image_array, 3, axis=-1)  # Convert to RGB by repeating the channel
        
        # Preprocess based on the model type
        if model_type == 'effnet':
            image_array = preprocess_input_effnet(image_array)
        elif model_type == 'vgg':
            
            image_array = preprocess_input_vgg(image_array)
        
        images.append(image_array)
    
    return np.vstack(images)