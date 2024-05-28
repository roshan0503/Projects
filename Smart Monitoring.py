import cv2
import mediapipe as mp
import pyttsx3
import os
import numpy as np

# Initialize MediaPipe Hands, Face Detection, and Text-to-Speech engine
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
engine = pyttsx3.init()

# Load images of students
student_images_path = "student_images"
student_images = {}
for file_name in os.listdir(student_images_path):
    name, ext = os.path.splitext(file_name)
    if ext.lower() in ['.jpg', '.jpeg', '.png']:
        student_images[name] = cv2.imread(os.path.join(student_images_path, file_name))

# Load Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Default name for unrecognized faces
default_name = "Unknown"

# Function to detect hand gestures and raise voice command
def detect_gesture(frame):
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        # Extract hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            # Assuming the hand is raised if certain landmarks are above a threshold
            if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < 0.7:
                # Prompt for student's name
                student_name = recognize_student(frame)
                if student_name != default_name:
                    # Send voice command to alert faculty
                    engine.say(f"{student_name} has raised a hand.")
                else:
                    engine.say("Someone has raised a hand.")
                engine.runAndWait()

    return frame

# Function to recognize student face from images
def recognize_student(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = gray_frame[y:y+h, x:x+w]

        # Compare with student images
        for name, image in student_images.items():
            # Resize image for comparison
            resized_image = cv2.resize(image, (w, h))
            resized_image_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

            # Compare faces using OpenCV's face recognizer
            # You may need a more sophisticated face recognition approach depending on your dataset size and quality
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train([face_roi], np.array([0]))  # Train with a single image for simplicity
            predicted_label, _ = recognizer.predict(resized_image_gray)

            # Assuming label 0 corresponds to the student's face
            if predicted_label == 0:
                return name

    return default_name  # Return default name if no match is found

# Main code for video capture and detection
cap = cv2.VideoCapture(0)  # Adjust the camera index as needed

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Face Detection
    face_results = face_detection.process(rgb_frame)

    num_faces = 0  # Counter for number of detected faces

    if face_results.detections:
        # Get the first detected face
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)

            # Draw bounding box around the face
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)

            # Increment face counter
            num_faces += 1

    # Trigger voice command for discussion if two or more faces are detected
    if num_faces >= 2:
        engine.say("There is a discussion in progress.")
        engine.runAndWait()

    # Call the detect_gesture function for hand detection
    frame = detect_gesture(frame)

    # Display the video frame with any additional visualizations
    cv2.imshow('Hand and Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
