import cv2

from deepface import DeepFace
import time

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_skip = 5  # Process emotion every 5 frames
frame_count = 0

# Initialize dominant attributes
dominant_emotion, dominant_gender, dominant_age = '', '', ''

# Timer for age and gender update
last_age_gender_update = time.time()
age_gender_interval = 15  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Process emotion every 5 frames
    if frame_count % frame_skip == 0:
        try:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            # Update age and gender every 15 seconds
            if current_time - last_age_gender_update > age_gender_interval:
                result = DeepFace.analyze(
                
                    small_frame,
                    actions=['emotion', 'age'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                dominant_gender = result[0]['dominant_gender']
                dominant_age = str(result[0]['age'])
                #dominant_emotion = result[0]['dominant_emotion']
                last_age_gender_update = current_time  # Reset timer

            else:
                # Only update emotion
                result = DeepFace.analyze(
                    small_frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                dominant_emotion = result[0]['dominant_emotion']

        except Exception as e:
            print("Error in DeepFace:", e)

    # Display results on the frame
    cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #qcv2.putText(frame, f"Gender: {dominant_gender}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Age: {dominant_age}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Optimized Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
