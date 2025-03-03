import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage import io, color

# Function to crop the brain contour from an image
def crop_brain_contour(image):
    """
    Crop the largest contour (brain) from the input grayscale image.

    Args:
    - image (numpy.ndarray): Grayscale image.

    Returns:
    - Cropped brain image (numpy.ndarray).
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Canny Edge Detection Algorithm
    thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours and grab the largest one
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        return image[y:y+h, x:x+w]  # Crop and return

    return image  # If no contour found, return original image

# Define preprocessing function
def preprocess_images(image_path):
    """
    Load, crop, resize, and preprocess image.

    Args:
    - image_path (str): path to image.

    Returns:
    - torch.Tensor: Preprocessed images as a batch tensor.
    """
    
    
    
    try:
        # Load image and convert to grayscale
        image = io.imread(image_path)
        gray = color.rgb2gray(image)
        gray_image = (gray * 255).astype(np.uint8)

        # Crop brain contour
        cropped_image = crop_brain_contour(gray_image)

        # Convert to PIL Image
        cropped_pil = Image.fromarray(cropped_image)

        # Define preprocessing transforms
        preprocess = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize to match model input
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]) # Normalization
        ])

        # Apply transformations
        image_tensor = preprocess(cropped_pil)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


    # Stack images into a batch tensor
    return image_tensor
