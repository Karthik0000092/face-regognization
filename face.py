import cv2
import numpy as np

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained_model.yml')

# Load the image to be recognized
img = cv2.imread('test_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Perform face recognition for each face found
for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    id_, confidence = recognizer.predict(roi_gray)
    
    if confidence < 70:  # You can adjust the threshold as needed
        # Get the label (name) of the recognized face
        # Assuming you have a dictionary mapping ids to names
        label = "Person " + str(id_)
    else:
        label = "Unknown"
    
    # Draw a rectangle around the face
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Display the image with faces recognized
cv2.imshow('Face Recognition', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
