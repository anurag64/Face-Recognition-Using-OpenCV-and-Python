import cv2
import numpy as np
import os
import pickle
from keras.models import load_model

# ==== CONFIGURATION ====
USE_DNN = False  # Set True to use DNN, False for Haarcascade
CONFIDENCE_THRESHOLD = 0.9  # Confidence below this will be treated as "Unknown"

# ==== Load Model and Labels ====
model = load_model("final_model.h5")     # Load the trained face recognition model
with open("clean data/label_classes.p", "rb") as f:
    label_classes = pickle.load(f)      # Load label to name mapping from file

# Function to return label name if confidence is high enough
def get_pred_label(pred, confidence):
    if confidence < CONFIDENCE_THRESHOLD:
        return "Unknown"                  # Return "Unknown" if confidence is too low
    return label_classes[pred] if pred < len(label_classes) else "Unknown"

# Function to preprocess the face image for model input
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (100, 100))     # Resize to model input size
    img = cv2.equalizeHist(img)  # Improve contrast
    img = img.reshape(1, 100, 100, 1)  # Reshape to 4D tensor (batch, height, width, channels)
    return img / 255.0    # Normalize pixel values

# ==== Load Face Detector ====
base_dir = os.path.dirname(os.path.abspath(__file__))    # Get current file directory

if USE_DNN:
    print("[INFO] Using DNN Face Detector")
    # Load DNN-based face detector (Caffe model)
    modelFile = os.path.join(base_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    configFile = os.path.join(base_dir, "deploy.prototxt")
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    print("[INFO] Using Haarcascade Face Detector")
    haar_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(haar_path)

# ==== Load Age & Gender Models ====
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']   # Predefined age groups
GENDER_LIST = ['Male', 'Female'] # Gender Classes

# Load pre-trained age prediction model
age_net = cv2.dnn.readNetFromCaffe(
    os.path.join(base_dir, "age_deploy.prototxt"),
    os.path.join(base_dir, "age_net.caffemodel"))

# Load pre-trained gender prediction model
gender_net = cv2.dnn.readNetFromCaffe(
    os.path.join(base_dir, "gender_deploy.prototxt"),
    os.path.join(base_dir, "gender_net.caffemodel"))

# ==== Start Webcam ====
cap = cv2.VideoCapture(0)  # Load pre-trained age prediction model
print("[INFO] Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break        # Exit loop if frame not read correctly

    faces = []     # List to store detected face bounding boxes

    if USE_DNN:
        # DNN-based face detection
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                faces.append((x1, y1, x2 - x1, y2 - y1))
    else:
        # Haarcascade-based face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # Covert to Greyscale
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

    # Process each detected face
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]   # Extract face region from frame
        if face.size == 0:
            continue     # Skip if face crop failed

        # ======== FACE RECOGNITION ========
        processed = preprocess(face)    # Preprocess face
        prediction = model.predict(processed)   # Predict class probabilities
        pred_index = np.argmax(prediction)   # Get class with highest probability
        confidence = np.max(prediction)    # Get the confidence score
        name_label = get_pred_label(pred_index, confidence)    # Get name or "Unknown"

        # ======== AGE & GENDER ========
        face_blob = cv2.dnn.blobFromImage(cv2.resize(face, (227, 227)), 1.0,
                                          (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        gender_net.setInput(face_blob)
        gender = GENDER_LIST[np.argmax(gender_net.forward())]    # Predict gender

        age_net.setInput(face_blob)
        age = AGE_BUCKETS[np.argmax(age_net.forward())]   # Predict age group

        full_label = f"{name_label}, {gender}, {age}"   # Combine all predictions

        # Draw rectangle around face and put label text above it
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, full_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face Recognition + Age/Gender", frame)      # Display Result
    if cv2.waitKey(1) == ord("q"):     # Quit if 'q' is pressed
        break

cap.release()    # Release webcam
cv2.destroyAllWindows()   # Close openCV Windows