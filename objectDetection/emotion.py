import cv2
from deepface import DeepFace

# Start capturing from the webcam
cap = cv2.VideoCapture(0)

# Emotion labels (predefined by DeepFace)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Detect faces and emotions using DeepFace
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Get the dominant emotion
        dominant_emotion = result[0]['dominant_emotion']

        # Display the emotion text on the image
        cv2.putText(frame, dominant_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print("Error in DeepFace:", e)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
