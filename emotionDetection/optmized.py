import cv2
from deepface import DeepFace
TF_ENABLE_ONEDNN_OPTS=0
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_skip = 5
frame_count = 0

dominant_emotion, dominant_gender, dominant_age = '', '', ''

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process every 5th frame
    if frame_count % frame_skip == 0:
        try:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            result = DeepFace.analyze(
                small_frame, 
                actions=['emotion', 'age'], 
                enforce_detection=False, 
                detector_backend='opencv'
            )

            dominant_emotion = result[0]['dominant_emotion']
            #dominant_gender = result[0]['dominant_gender']
            dominant_age = str(result[0]['age'])

        except Exception as e:
            print("Error in DeepFace:", e)

    # Show cached results
    cv2.putText(frame, f"emotion:{dominant_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #cv2.putText(frame, f"gender:{dominant_gender}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"age:{dominant_age}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
