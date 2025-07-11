import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from collections import Counter

# Directories
data_dir = "E:\\Face Recognition Project\\AIML project\\clean data"   # Folder to store processed data
img_dir = "E:\\Face Recognition Project\\AIML project\\images"   # Folder containing raw images

# Ensure clean data directory exists
os.makedirs(data_dir, exist_ok=True)

#==== Initialize lists to store image arrays and labels ====
image_data = []
labels = []

# ==== Loop through all files in the image directory ====
for filename in os.listdir(img_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue  # Skip non-image files

    image_path = os.path.join(img_dir, filename)   # Full image path
    image = cv2.imread(image_path)       # Read Image Using OpenCV

    if image is None:
        print(f"Skipping unreadable image: {filename}")
        continue    # Skip corrupted/unreadable images


    # Resize image to 100x100 and convert to grayscale
    image = cv2.resize(image, (100, 100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_data.append(image)  # Add to image dataset


    # Extract label (e.g., "Anurag_0.jpg" → "Anurag")
    label = filename.split("_")[0]
    labels.append(label)  # Add to labels list

# ==== Convert image data to NumPy array, normalize, and reshape ====
image_data = np.array(image_data).reshape(-1, 100, 100, 1) / 255.0
labels = np.array(labels)   # ==== Convert labels to NumPy array ====

# ==== Encode string labels to integers ====
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# ==== Convert labels to one-hot encoded format (for multiclass training) ====
labels_categorical = to_categorical(labels_encoded, num_classes=len(le.classes_)) 

# ==== Save preprocessed images ====
with open(os.path.join(data_dir, "images.p"), 'wb') as f:
    pickle.dump(image_data, f)

# ==== Save one-hot encoded labels ====
with open(os.path.join(data_dir, "labels.p"), 'wb') as f:
    pickle.dump(labels_categorical, f)

# ==== Save class names (e.g., ['Anurag', 'Aniket']) for future decoding ====
with open(os.path.join(data_dir, "label_classes.p"), 'wb') as f:
    pickle.dump(le.classes_, f)

# ==== Print debug info ====
print("Data saved successfully!")
print(f" - images.p shape: {image_data.shape}")
print(f" - labels.p shape: {labels_categorical.shape}")
print(f" - Classes: {list(le.classes_)}")

# ==== Display one sample image from the dataset ====
plt.imshow(image_data[0].reshape(100, 100), cmap="gray")
plt.title(f"Label: {labels[0]}")
plt.show()

# ==== Print count of images per label/person ====
print("\n Image count per person:")
print(Counter(labels))

# ==== Print mapping of label name to integer index ====
print("\nLabel → Index Mapping:")
for index, name in enumerate(le.classes_):
    print(f"  {name} → {index}")