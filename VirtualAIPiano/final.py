import pygame
import time
import cv2
import mediapipe as mp
from dataclasses import dataclass
from typing import Set, Tuple

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

@dataclass
class StairZone:
    x_range: range
    y_range: range
    coordinates: Set[Tuple[int, int]]
    sound: str

# Replace the coordinate lists with more efficient structures
def create_stair_zone(x_start, x_end, y_start, y_end, sound):
    coords = set()
    x_range = range(x_start, x_end + 1)
    y_range = range(y_start, y_end + 1)
    for x in x_range:
        for y in y_range:
            coords.add((x, y))
    return StairZone(x_range, y_range, coords, sound)

# Define stair zones more efficiently
ankle_zones = [
    create_stair_zone(288, 293, 288, 293, 'a4.mp3'),
    create_stair_zone(299, 305, 299, 305, 'b4.mp3'),
    create_stair_zone(306, 310, 306, 310, 'c4.mp3'),
    create_stair_zone(315, 320, 315, 320, 'd4.mp3'),
    create_stair_zone(324, 332, 324, 332, 'e4.mp3')
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
cap = cv2.VideoCapture(0)

class SoundManager:
    def __init__(self):
        self._sound_cache = {}
        self._last_played_sound = None
        self._last_play_time = 0
        self.MIN_INTERVAL = 0.1  # Minimum seconds between sounds

    def preload_sounds(self, sound_files):
        for sound_file in sound_files:
            try:
                self._sound_cache[sound_file] = pygame.mixer.Sound(sound_file)
            except Exception as e:
                print(f"Error loading sound {sound_file}: {e}")

    def play_sound(self, sound_file):
        current_time = time.time()
        if (sound_file != self._last_played_sound and 
            current_time - self._last_play_time >= self.MIN_INTERVAL):
            if sound_file not in self._sound_cache:
                self.preload_sounds([sound_file])
            
            self._sound_cache[sound_file].play()
            self._last_played_sound = sound_file
            self._last_play_time = current_time

# Initialize sound manager
sound_manager = SoundManager()
sound_manager.preload_sounds([zone.sound for zone in ankle_zones])

def check_ankle_position(ankle_coords, zones):
    x, y = ankle_coords
    for zone in zones:
        if (x in zone.x_range and y in zone.y_range and 
            (x, y) in zone.coordinates):
            return zone.sound
    return None

# Main loop improvements
frame_count = 0
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame_count += 1
    if frame_count % PROCESS_EVERY_N_FRAMES != 0:
        continue

    # Convert and process frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        height, width, _ = frame.shape
        
        # Get ankle coordinates more efficiently
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        left_ankle_coords = (
            int(left_ankle.y * height),
            int(left_ankle.x * width)
        )
        right_ankle_coords = (
            int(right_ankle.y * height),
            int(right_ankle.x * width)
        )

        # Check if ankles are in valid range
        if (y_min <= left_ankle_coords[0] <= y_max and 
            y_min <= right_ankle_coords[0] <= y_max):
            
            # Check for matching zones
            sound_file = check_ankle_position(left_ankle_coords, ankle_zones)
            if sound_file:
                sound_manager.play_sound(sound_file)

    # Display frame and check for exit
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
