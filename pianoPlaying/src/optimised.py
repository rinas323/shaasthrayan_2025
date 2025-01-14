import cv2
import mediapipe as mp
import pygame
import time
from collections import deque

# Initialize Pygame Mixer for sound
pygame.mixer.init()

# Preload sounds
sounds = {
    "C.wav": pygame.mixer.Sound("b4.mp3"),
    "D.wav": pygame.mixer.Sound("c4.mp3"),
    "E.wav": pygame.mixer.Sound("d4.mp3"),
}

# Define staircase regions
stair_steps = [
    {"y_min": 310, "y_max": 260, "sound": "C.wav"},
    {"y_min": 320, "y_max": 310, "sound": "D.wav"},
    {"y_min": 330, "y_max": 320, "sound": "E.wav"},
]

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Variables for debounce
last_played_step = None
cooldown = 0.5  # 0.5 seconds cooldown
last_played_time = time.time()

# Smoothing foot detection
left_y_history = deque(maxlen=5)
right_y_history = deque(maxlen=5)

def play_sound(sound_file):
    sounds[sound_file].play()

def smooth_position(y, history):
    history.append(y)
    return sum(history) / len(history)

def check_step(y_coord):
    global last_played_step, last_played_time
    current_time = time.time()
    for idx, step in enumerate(stair_steps):
        if step["y_min"] <= y_coord <= step["y_max"]:
            if last_played_step != idx or (current_time - last_played_time) > cooldown:
                play_sound(step["sound"])
                last_played_step = idx
                last_played_time = current_time

def draw_steps(frame, width):
    for idx, step in enumerate(stair_steps):
        color = (0, 255, 0) if last_played_step == idx else (255, 0, 0)
        cv2.rectangle(frame, (0, step["y_min"]), (width, step["y_max"]), color, 2)

fps = 30
prev_time = 0

while cap.isOpened():
    current_time = time.time()
    if current_time - prev_time >= 1 / fps:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        height, width, _ = frame.shape

        if results.pose_landmarks:
            left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

            left_y = int(left_ankle.y * height)
            right_y = int(right_ankle.y * height)

            smoothed_left_y = smooth_position(left_y, left_y_history)
            smoothed_right_y = smooth_position(right_y, right_y_history)

            check_step(smoothed_left_y)
            check_step(smoothed_right_y)

        draw_steps(frame, width)
        cv2.imshow('Staircase Piano', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        prev_time = current_time

cap.release()
cv2.destroyAllWindows()
