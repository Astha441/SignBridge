import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

gesture_map = {
    0: "Yes",
    1: "hello",
    2: "I love you"
}

# Initialize OpenCV video capture object
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands solution
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    ret, frame = cap.read()

    # Flip the frame horizontally for a more natural selfie-view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using MediaPipe hands solution
    results = hands.process(image)

    if results.multi_hand_landmarks:
        # Draw hand landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # Get the hand-landmark data
        landmarks = results.multi_hand_landmarks[0].landmark

        # Extract the x, y, z coordinates of the tip of the index finger
        x, y, z = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].z

        # Calculate the distance between the tip of the index finger and the wrist
        distance = np.linalg.norm(np.array([landmarks[mp_hands.HandLandmark.WRIST].x, landmarks[mp_hands.HandLandmark.WRIST].y, landmarks[mp_hands.HandLandmark.WRIST].z]) - np.array([x, y, z]))

        # Classify the gesture based on the distance and the angle between the index finger and the thumb
        if distance > 0.15 and x > 0.4:
            if (landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y - landmarks[mp_hands.HandLandmark.THUMB_TIP].y) > 0.05:
                gesture = gesture_map[0]
            elif (landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x - landmarks[mp_hands.HandLandmark.THUMB_TIP].x) > 0.05:
                gesture = gesture_map[1]
            else:
                gesture = gesture_map[2]

            # Display the recognized gesture on the frame
            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()