import pygame
import cv2
import mediapipe as mp
import time
from dataclasses import dataclass
from typing import Set, Tuple, List
import logging
from queue import Queue
import threading
from collections import deque

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Debug mode
DEBUG = True

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class CalibrationSystem:
    def __init__(self):
        self.calibrated = False
        self.calibration_frames = []
        self.calibration_points = []
        self.reference_height = None
        self.scale_factor = 1.0
        self.offset = (0, 0)
        
    def collect_calibration_frame(self, landmarks):
        if len(self.calibration_frames) < 30:  # Collect 30 frames for stability
            if landmarks.pose_landmarks:
                # Get ankle positions
                left_ankle = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
                # Get hip positions for scale reference
                left_hip = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                
                self.calibration_frames.append({
                    'ankles': (left_ankle, right_ankle),
                    'hips': (left_hip, right_hip)
                })
            return False
        else:
            self._process_calibration()
            return True
            
    def _process_calibration(self):
        # Calculate average positions and body proportions
        avg_hip_height = 0
        avg_ankle_pos = {'x': 0, 'y': 0}
        
        for frame in self.calibration_frames:
            left_ankle, right_ankle = frame['ankles']
            left_hip, right_hip = frame['hips']
            
            # Calculate average hip height for scale reference
            hip_height = abs(left_hip.y - right_hip.y)
            avg_hip_height += hip_height
            
            # Calculate average ankle position
            ankle_x = (left_ankle.x + right_ankle.x) / 2
            ankle_y = (left_ankle.y + right_ankle.y) / 2
            avg_ankle_pos['x'] += ankle_x
            avg_ankle_pos['y'] += ankle_y
        
        n_frames = len(self.calibration_frames)
        avg_hip_height /= n_frames
        avg_ankle_pos['x'] /= n_frames
        avg_ankle_pos['y'] /= n_frames
        
        # Set reference values
        self.reference_height = avg_hip_height
        self.scale_factor = 100 / avg_hip_height  # Normalize to 100 units
        self.offset = (avg_ankle_pos['x'], avg_ankle_pos['y'])
        self.calibrated = True
        
    def transform_coordinates(self, x, y):
        if not self.calibrated:
            return x, y
            
        # Apply scale and offset transformation
        adjusted_x = (x - self.offset[0]) * self.scale_factor
        adjusted_y = (y - self.offset[1]) * self.scale_factor
        return adjusted_x, adjusted_y

class SoundManager:
    def __init__(self):
        self._sound_cache = {}
        self._last_played_sound = None
        self._current_step_sound = None
        
    def preload_sounds(self, sound_files):
        for sound_file in sound_files:
            try:
                self._sound_cache[sound_file] = pygame.mixer.Sound(sound_file)
            except Exception as e:
                logger.error(f"Error loading sound {sound_file}: {e}")
        
    def play_sound(self, sound_file):
        # Only play if it's a different step than current
        if sound_file != self._current_step_sound:
            if sound_file not in self._sound_cache:
                self._sound_cache[sound_file] = pygame.mixer.Sound(sound_file)
            
            self._sound_cache[sound_file].play()
            self._current_step_sound = sound_file

class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.frame_times = deque(maxlen=window_size)
        self.last_frame_time = None

    def frame_start(self):
        self.last_frame_time = time.perf_counter()

    def frame_end(self):
        if self.last_frame_time:
            self.frame_times.append(time.perf_counter() - self.last_frame_time)

    def get_fps(self):
        if not self.frame_times:
            return 0
        return len(self.frame_times) / sum(self.frame_times)

@dataclass
class StairZone:
    relative_x_range: range
    relative_y_range: range
    sound: str

# Define zones in relative coordinates (0-100 scale)
ankle_zones = [
    StairZone(range(20, 31), range(20, 31), 'a4.mp3'),
    StairZone(range(35, 46), range(35, 46), 'b4.mp3'),
    StairZone(range(50, 61), range(50, 61), 'c4.mp3'),
    StairZone(range(65, 76), range(65, 76), 'd4.mp3'),
    StairZone(range(80, 91), range(80, 91), 'e4.mp3')
]

def process_landmarks(pose_landmarks, frame, calibration):
    height, width, _ = frame.shape
    
    # Get ankle coordinates
    left_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    
    # Transform coordinates using calibration
    left_x, left_y = calibration.transform_coordinates(left_ankle.x, left_ankle.y)
    right_x, right_y = calibration.transform_coordinates(right_ankle.x, right_ankle.y)
    
    # Check zones using relative coordinates
    for zone in ankle_zones:
        if (int(left_x) in zone.relative_x_range and int(left_y) in zone.relative_y_range) or \
           (int(right_x) in zone.relative_x_range and int(right_y) in zone.relative_y_range):
            sound_manager.play_sound(zone.sound)
            if DEBUG:
                cv2.circle(frame, 
                          (int(left_ankle.x * width), int(left_ankle.y * height)), 
                          10, (0, 255, 0), -1)
            break

def draw_debug_info(frame, monitor, calibration):
    if not calibration.calibrated:
        frames_remaining = 30 - len(calibration.calibration_frames)
        cv2.putText(
            frame,
            f"Calibrating... Please stand still ({frames_remaining} frames remaining)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            frame,
            f"FPS: {monitor.get_fps():.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

@dataclass
class StairStep:
    coordinates: Tuple[float, float]  # (x, y) coordinates
    sound: str

class StairCalibration:
    def __init__(self):
        self.steps = []
        self.current_step = 0
        self.total_steps = 6
        # Added f4.mp3 as the sixth sound
        self.sound_files = ['a4.mp3', 'b4.mp3', 'c4.mp3', 'd4.mp3', 'e4.mp3', 'f4.mp3']
        
    def capture_step(self, x, y):
        if self.current_step < self.total_steps:
            sound_file = self.sound_files[self.current_step]
            self.steps.append(StairStep((x, y), sound_file))
            self.current_step += 1
            return True
        return False

    def is_complete(self):
        return self.current_step >= self.total_steps

def check_position_match(results, step_coords, tolerance=0.05):
    left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    
    current_x = (left_ankle.x + right_ankle.x) / 2
    current_y = (left_ankle.y + right_ankle.y) / 2
    
    return (abs(current_x - step_coords[0]) < tolerance and 
            abs(current_y - step_coords[1]) < tolerance)

def main():
    # Initialize systems
    pygame.mixer.init()
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Initialize camera (try different indices if 0 doesn't work)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Error: Could not open camera")
        return

    calibration = CalibrationSystem()
    global sound_manager
    sound_manager = SoundManager()
    sound_manager.preload_sounds([zone.sound for zone in ankle_zones])
    performance_monitor = PerformanceMonitor()
    
    logger.info("Starting calibration process...")
    logger.info("Please stand in view of the camera and remain still...")
    
    stair_calibration = StairCalibration()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Draw the pose landmarks
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            if not stair_calibration.is_complete():
                # Calibration mode
                left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
                
                x = (left_ankle.x + right_ankle.x) / 2
                y = (left_ankle.y + right_ankle.y) / 2
                
                # Show calibration status
                cv2.putText(frame, 
                    f"Stand on step {stair_calibration.current_step + 1} and press 's'", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    stair_calibration.capture_step(x, y)
                    print(f"Step {stair_calibration.current_step} captured!")
            else:
                # Detection mode
                for step in stair_calibration.steps:
                    if check_position_match(results, step.coordinates):
                        sound_manager.play_sound(step.sound)
        
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
