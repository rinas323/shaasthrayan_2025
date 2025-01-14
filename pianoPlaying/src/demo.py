import cv2

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Define the six stair rectangles using start (x1, y1) and end (x2, y2) coordinates
stair_steps = [
    {"start": (844, 497), "end": (525, 507)},  # Stair 1
    {"start": (848, 511), "end": (508, 523)},  # Stair 2
    {"start": (857, 521), "end": (507, 533)},  # Stair 3
    {"start": (867, 535), "end": (487, 547)},  # Stair 4
    {"start": (878, 548), "end": (471, 564)},  # Stair 5
    {"start": (890, 563), "end": (457, 582)}   # Stair 6
]

# Loop to capture video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Draw the six stair rectangles on the frame
    for stair in stair_steps:
        start_point = stair["start"]
        end_point = stair["end"]
        cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)  # Green rectangles

    # Display the frame
    cv2.imshow('Staircase Rectangles', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
