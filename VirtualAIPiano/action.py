import pygame
import time
import cv2
import mediapipe as mp

# Initialize pygame mixer
pygame.mixer.init()

# Define a function to play sound
def play_sound(sound_file):
    # Load a sound (make sure to provide a valid path to a sound file on your system)
    sound = pygame.mixer.Sound(sound_file)  # Use the corresponding sound file
    sound.play()

# Define the area to check (example values)
y_min = 200  # Minimum y value of the ankle area
y_max = 345  # Maximum y value of the ankle area

# Group 1: Coordinates for Staircase 1
ankle_data_staircase1 = [
        (288, 288), (288, 289), (288, 290), (288, 291), (288, 292), (288, 293), 
(289, 288), (289, 289), (289, 290), (289, 291), (289, 292), (289, 293), 
(290, 288), (290, 289), (290, 290), (290, 291), (290, 292), (290, 293), 
(291, 288), (291, 289), (291, 290), (291, 291), (291, 292), (291, 293), 
(292, 288), (292, 289), (292, 290), (292, 291), (292, 292), (292, 293), 
(293, 288), (293, 289), (293, 290), (293, 291), (293, 292), (293, 293)

]

# Group 2: Coordinates for Staircase 2
ankle_data_staircase2 = [
(299, 299), (299, 300), (299, 301), (299, 302), (299, 303), (299, 304), (299, 305),
(300, 299), (300, 300), (300, 301), (300, 302), (300, 303), (300, 304), (300, 305),
(301, 299), (301, 300), (301, 301), (301, 302), (301, 303), (301, 304), (301, 305),
(302, 299), (302, 300), (302, 301), (302, 302), (302, 303), (302, 304), (302, 305),
(303, 299), (303, 300), (303, 301), (303, 302), (303, 303), (303, 304), (303, 305),
(304, 299), (304, 300), (304, 301), (304, 302), (304, 303), (304, 304), (304, 305),
(305, 299), (305, 300), (305, 301), (305, 302), (305, 303), (305, 304), (305, 305)

]

# Group 3: Coordinates for Staircase 3
ankle_data_staircase3 = [
 (306, 306), (306, 307), (306, 308), (306, 309), (306, 310),
(307, 306), (307, 307), (307, 308), (307, 309), (307, 310),
(308, 306), (308, 307), (308, 308), (308, 309), (308, 310),
(309, 306), (309, 307), (309, 308), (309, 309), (309, 310),
(310, 306), (310, 307), (310, 308), (310, 309), (310, 310)

]

# Group 4: Coordinates for Staircase 4
ankle_data_staircase4 = [
(315, 315), (315, 316), (315, 317), (315, 318), (315, 319), (315, 320),
(316, 315), (316, 316), (316, 317), (316, 318), (316, 319), (316, 320),
(317, 315), (317, 316), (317, 317), (317, 318), (317, 319), (317, 320),
(318, 315), (318, 316), (318, 317), (318, 318), (318, 319), (318, 320),
(319, 315), (319, 316), (319, 317), (319, 318), (319, 319), (319, 320),
(320, 315), (320, 316), (320, 317), (320, 318), (320, 319), (320, 320)

]

# Group 5: Coordinates for Staircase 5
ankle_data_staircase5 = [
(324, 324), (324, 325), (324, 326), (324, 327), (324, 328), (324, 329), (324, 330), (324, 331), (324, 332),
(325, 324), (325, 325), (325, 326), (325, 327), (325, 328), (325, 329), (325, 330), (325, 331), (325, 332),
(326, 324), (326, 325), (326, 326), (326, 327), (326, 328), (326, 329), (326, 330), (326, 331), (326, 332),
(327, 324), (327, 325), (327, 326), (327, 327), (327, 328), (327, 329), (327, 330), (327, 331), (327, 332),
(328, 324), (328, 325), (328, 326), (328, 327), (328, 328), (328, 329), (328, 330), (328, 331), (328, 332),
(329, 324), (329, 325), (329, 326), (329, 327), (329, 328), (329, 329), (329, 330), (329, 331), (329, 332),
(330, 324), (330, 325), (330, 326), (330, 327), (330, 328), (330, 329), (330, 330), (330, 331), (330, 332),
(331, 324), (331, 325), (331, 326), (331, 327), (331, 328), (331, 329), (331, 330), (331, 331), (331, 332),
(332, 324), (332, 325), (332, 326), (332, 327), (332, 328), (332, 329), (332, 330), (332, 331), (332, 332)

]

# List of groups with their corresponding sounds
ankle_groups = [
    {'coordinates': ankle_data_staircase1, 'sound': 'a4.mp3'},  # Replace with your sound file paths
    {'coordinates': ankle_data_staircase2, 'sound': 'b4.mp3'},
    {'coordinates': ankle_data_staircase3, 'sound': 'c4.mp3'},
    {'coordinates': ankle_data_staircase4, 'sound': 'd4.mp3'},
    {'coordinates': ankle_data_staircase5, 'sound': 'e4.mp3'}
]

# Variable to keep track of the last played sound
last_played_sound = None

# Function to play the sound only if it's different from the last one
def play_stair_sound(sound):
    global last_played_sound
    if sound != last_played_sound:
        play_sound(sound)
        last_played_sound = sound
    else:
        print(f"Sound {sound} already played, skipping.")

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Set up webcam
cap = cv2.VideoCapture(1)

# Main loop to capture frames and process them
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with the Pose model
    results = pose.process(rgb_frame)

    # Check if landmarks were detected
    if results.pose_landmarks:
        # Draw the pose landmarks on the frame (optional, for visualization)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get the y-coordinates of the left and right ankles (Landmarks 29 and 30 for left and right ankles)
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # Convert normalized coordinates to pixel coordinates
        height, width, _ = frame.shape
        left_ankle_y = int(left_ankle.y * height)
        right_ankle_y = int(right_ankle.y * height)

        # Print ankle coordinates (optional)
        print(f"Left ankle y: {left_ankle_y}, Right ankle y: {right_ankle_y}")

        # Check if both ankles' y-values are within the defined area
        if y_min <= left_ankle_y <= y_max and y_min <= right_ankle_y <= y_max:
            # Check each group of ankle coordinates to see if there's a match
            for group in ankle_groups:
                if (left_ankle_y, right_ankle_y) in group['coordinates']:
                    print(f"Ankle coordinates match for {group['sound']}! Playing sound.")
                    play_stair_sound(group['sound'])  # Play sound only if it's different from the last one
                    time.sleep(1)  # Wait 1 second before checking the next values

    # Display the captured frame in a window (optional)
    cv2.imshow("Webcam", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
