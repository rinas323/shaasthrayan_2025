import cv2

# Initialize variables
drawing = False  # True if mouse is pressed
start_point = (-1, -1)
end_point = (-1, -1)
rectangles = []

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing, rectangles

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        rectangles.append((start_point, end_point))
        print(f"Rectangle Coordinates: Start {start_point}, End {end_point}")

# Load the image where you want to draw the rectangles
img = cv2.imread("stair6.jpg")  # Replace with your image file path
if img is None:
    print("Error loading image. Check the file path.")
    exit()

clone = img.copy()
cv2.namedWindow("Draw Rectangle")
cv2.setMouseCallback("Draw Rectangle", draw_rectangle)

while True:
    temp_img = clone.copy()

    # Draw all saved rectangles
    for rect in rectangles:
        cv2.rectangle(temp_img, rect[0], rect[1], (0, 255, 0), 2)

    # Draw the current rectangle while dragging
    if drawing and start_point != (-1, -1) and end_point != (-1, -1):
        cv2.rectangle(temp_img, start_point, end_point, (0, 0, 255), 2)

    cv2.imshow("Draw Rectangle", temp_img)

    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to exit
    if key == ord('q'):
        break
    # Press 'c' to clear all rectangles
    elif key == ord('c'):
        rectangles = []
        clone = img.copy()
        print("Cleared all rectangles.")

cv2.destroyAllWindows()

# Output all rectangle coordinates
print("\nAll Selected Rectangle Coordinates:")
for idx, rect in enumerate(rectangles):
    print(f"Rectangle {idx + 1}: Start {rect[0]}, End {rect[1]}")
