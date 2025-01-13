import cv2
import numpy as np
import mediapipe as mp
import pygame  # For playing sound


# Initialize pygame mixer for sound
pygame.mixer.init()

# Mapping fingers to piano notes
note_files = {
    "index": "b4.mp3",  # Play note_1 for index finger raised
    "middle": "c4.mp3",  # Play note_2 for middle finger raised
    "ring": "d4.mp3",    # Play note_3 for ring finger raised
    "pinky": "e4.mp3",   # Play note_4 for pinky finger raised
    "thumb": "f4.mp3",   # Play note_5 for thumb raised
}
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing for landmarks visualization
mp_drawing = mp.solutions.drawing_utils

# Track the previous state of each finger (whether raised or lowered)
finger_state = {
    "index": False,  # Initially not raised
    "middle": False,
    "ring": False,
    "pinky": False,
    "thumb": False,
}

# Function to play a specific piano note
def play_piano_note(note_name):
    """Function to play a piano note based on the finger raised."""
    if note_name in note_files:
        pygame.mixer.music.load(note_files[note_name])
        pygame.mixer.music.play()

# Initialize video capture (0 for the default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for hand detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform hand detection
    results = hands.process(rgb_frame)

    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check if a finger is raised
            for finger, idx in [("index", 8), ("middle", 12), ("ring", 16), ("pinky", 20), ("thumb", 4)]:
                # Get the landmark for the finger tip (tip of the finger is the one with the highest index)
                x = hand_landmarks.landmark[idx].x
                y = hand_landmarks.landmark[idx].y

                # For simplicity, let's check if the y-coordinate of the fingertip is above a certain threshold
                # (indicating that the finger is raised)
                is_raised = y < 0.5  # Arbitrary threshold, can be adjusted

                # Check if the finger state has changed (from folded to raised)
                if is_raised and not finger_state[finger]:  # If the finger is raised and was previously not raised
                    print(f"{finger.capitalize()} Finger Raised!")
                    play_piano_note(finger)  # Play the corresponding note
                    finger_state[finger] = True  # Update the state to raised

                # If the finger is folded (y > 0.5), update the state to not raised
                if not is_raised and finger_state[finger]:  # If the finger is folded and was previously raised
                    finger_state[finger] = False  # Update the state to folded

    # Display the frame with landmarks and connections
    cv2.imshow('Piano Note Control with Finger Gestures', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()