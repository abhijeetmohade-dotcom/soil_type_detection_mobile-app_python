import os
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

def preprocess(img_path, target_size=(224,224)):
    img = Image.open(img_path).convert('RGB').resize(target_size)
    arr = np.array(img) / 255.0
    return arr.flatten().reshape(1, -1)

def predict_soil(test_img_path, data_dir='soil-data', target_size=(224,224)):
    best_score = -1
    best_class = None
    test_vec = preprocess(test_img_path, target_size)
    for soil_class in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, soil_class)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                train_vec = preprocess(img_path, target_size)
                score = cosine_similarity(test_vec, train_vec)[0][0]
                if score > best_score:
                    best_score = score
                    best_class = soil_class
    return best_class

# Usage:
img_path = "test_soil.jpg"  # Change to your soil image filename
soil_class = predict_soil(img_path)
print(f"Predicted class: {soil_class}")
