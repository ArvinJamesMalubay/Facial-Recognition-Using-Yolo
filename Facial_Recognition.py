import datetime
import cv2
import json
import numpy as np

# Paths to model and label map
model_path = "trainer.yml"  # Path to the trained model
label_map_path = "label_mapping.json"  # Path to the label mapping JSON file

# Load the trained model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(model_path)

# Load the label mapping from JSON
with open(label_map_path, "r") as file:
    labels = json.load(file)
labels = {v: k for k, v in labels.items()}

cap = cv2.VideoCapture(0)
threshold = 70

# Load a pre-trained deep learning face detector (e.g., using Dlib or any other face detector)
# Here, we'll use OpenCV's deep learning-based face detector (DNN module)
deploy = "face_detection_model/deploy.prototxt"
ressnet = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(deploy, ressnet)

# Initialize FPS counter
fps_start_time = datetime.datetime.now()
fps_frame_count = 0
print("Starting video stream. Press 'q' to exit.")
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
        # Calculate FPS
    fps_frame_count += 1
    fps_elapsed_time = (datetime.datetime.now() - fps_start_time).total_seconds()
    fps = fps_frame_count / fps_elapsed_time
    # Display FPS on top left corner of monitor
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    # Get the frame's height and width
    h, w = frame.shape[:2]

    # Convert the frame to a blob suitable for DNN-based face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Loop through detected faces (detections returned by DNN model)
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # If confidence is above threshold, process this face
        if confidence > 0.7:  # Threshold for confidence
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")

            # Extract the face region of interest (ROI)
            face_roi = frame[y:y2, x:x2]
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Predict the identity using the recognizer
            label_id, pred_confidence = face_recognizer.predict(gray_face)

            print(pred_confidence)

            # Apply the threshold: if confidence < threshold, set label to "Unknown"
            if pred_confidence < threshold or pred_confidence > 100:
                label = "Unknown"
            else:
                label = labels.get(label_id, "Unknown")

            # Display the label and confidence on the image
            text = f"{label} ({pred_confidence:.2f}%)"
            color = (255, 255, 0) if label != "Unknown" else (0, 0, 255)

            # Calculate the position of the text (above the face or below)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_width, text_height = text_size

            # Position the text above the face
            text_x = x + (x2 - x) // 2 - text_width // 2
            text_y = y - 10 if y - 10 > text_height else y2 + text_height + 10  # If the text goes off-screen above, put it below

            # Ensure the text doesn't go outside the frame
            if text_x < 0:
                text_x = 0
            elif text_x + text_width > frame.shape[1]:
                text_x = frame.shape[1] - text_width

            if text_y < 0:
                text_y = 0
            elif text_y + text_height > frame.shape[0]:
                text_y = frame.shape[0] - text_height

            # Draw a rectangle around the face and display the label
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


    # Display the frame
    cv2.imshow("Facial Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
