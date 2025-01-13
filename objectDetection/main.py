import cv2
import numpy as np

# Load YOLOv3 model and config files
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO dataset labels (YOLOv3 is trained on this dataset)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get the index of 'book' class from COCO dataset
book_class_id = classes.index('book')

# Get the output layers of the YOLO model
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the input image or video
cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    height, width, _ = frame.shape
    boxes, confidences, class_ids = [], [], []

    # Process each detection
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter for 'book' detection with confidence threshold
            if class_id == book_class_id and confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to remove duplicate boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Book Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
