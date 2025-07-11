import cv2              # OpenCV for computer vision functions
import numpy as np      # NumPy for array operations
import os               # OS for directory and path handling
import time             # Time module for delays

# Load Haar Cascade Classifier
base_dir = os.path.dirname(os.path.abspath(__file__))    # Get current script directory
cascade_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")  ## Path to Haar XML file
classifier = cv2.CascadeClassifier(cascade_path)         # Load the classifier

# Directory to save images
save_dir = "E:\\Face Recognition Project\\AIML project\\images"
os.makedirs(save_dir, exist_ok=True)         # Create directory if it doesn't exist

# Initialize webcam
cap = cv2.VideoCapture(0)      # Start webcam capture (0 for default camera)
data = []                      # List to store captured face images
count = 0                      # Counter for number of images captured
max_images = 100               # Maximum number of images to capture

print("[INFO] Starting webcam face capture...")

while count < max_images:
    ret, frame = cap.read()          # Read a frame from webcam
    if not ret:
        print("[ERROR] Failed to capture frame from webcam.")
        continue                     # Skip this iteration if frame capture failed

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_points = classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in face_points:    # Crop the face from the original colored frame
        face_frame = frame[y:y + h, x:x + w]
        face_frame = cv2.resize(face_frame, (100, 100))      # Resize the face to 100x100 pixels (standard input size)
        face_gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)   # Convert to grayscale before saving (as model expects grayscale input)
       
       # Add face image to the data list
        data.append(face_gray)
        count += 1

        print(f"[INFO] Captured {count}/{max_images}")
        cv2.imshow("Face", face_gray)    # Show the face being captured
        time.sleep(0.1)   # Small delay to avoid rapid-fire captures
        break            # Exit the inner loop after capturing one face per frame

    cv2.putText(frame, f"Capturing: {count}/{max_images}", (20, 40),  # Show webcam feed with capture progress overlayed
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Webcam Feed", frame)
 
    # Break loop if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quit requested.")
        break

cap.release()    # Stop webcam
cv2.destroyAllWindows() #Close all openCV windows

# Save images if enough were collected
if count == max_images:
    name = input("Enter the person's name: ").strip()   # Prompt user to input name
    for i in range(max_images):
        filename = os.path.join(save_dir, f"{name}_{i}.jpg")  # Generate filename with person's name
        cv2.imwrite(filename, data[i])      # Save face image
    print(f"[INFO] {max_images} face images saved for '{name}' in: {save_dir}")
else:
    print("[INFO] Not enough data collected. Only", count, "images were saved.")