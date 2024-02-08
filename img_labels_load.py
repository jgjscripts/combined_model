import os
from PIL import Image
import numpy as np

def load_images_and_labels(folder_path):
    images = []
    labels = []

    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                if image_path.endswith('.jpg') or image_path.endswith('.png'):  # Filter for image files
                    try:
                        image = Image.open(image_path)
                        images.append(np.array(image))  # Convert image to numpy array and append
                        labels.append(label)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")

    return images, labels

folder_path = "path_to_your_folder"
images, labels = load_images_and_labels(folder_path)

# Example usage: Print the number of loaded images and labels
print(f"Number of images: {len(images)}")
print(f"Number of labels: {len(labels)}")
