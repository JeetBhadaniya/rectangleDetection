from transform import four_point_transform
import cv2
import numpy as np

global capture_mode, selected_rect,indexReturn,contours
capture_mode = True
selected_rect = None
counter = 1


def mouse_callback(event, x, y, flags, param):
    global selected_rect, capture_mode

    if not capture_mode and selected_rect is not None:
        if event == cv2.EVENT_LBUTTONDOWN:
            for i in range(4):
                if np.linalg.norm((x, y) - selected_rect[i]) < 10:
                    selected_rect[i] = (x, y)
                    #capture_mode = False
                    break

# Function to find the biggest rectangle
def biggestRectangle(contours):
    biggest = None
    max_area = 0
    indexReturn = -1
    for index in range(len(contours)):
        i = contours[index]
        area = cv2.contourArea(i)
        if area > 100:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.1 * peri, True)
            if area > max_area:
                biggest = approx
                max_area = area
                indexReturn = index
    return indexReturn, biggest



# Variables to store the four corner points of the rectangle
#selected_rect = np.zeros((4, 2), dtype=np.float32)

# Initialize the camera capture object
cap = cv2.VideoCapture(0)

# Create two windows
#cv2.namedWindow('Camera Feed')
cv2.namedWindow("Document Scanner2")
cv2.setMouseCallback("Document Scanner2", mouse_callback)

while True:
    key = cv2.waitKey(1) & 0xFF
    if capture_mode:
        # Capture a frame from the camera
        ret, frame = cap.read()

        frame2 = frame.copy()

        # Check if the frame was captured successfully
        if not ret:
            print("Failed to capture frame")
            break

        # Convert the captured frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to the grayscale image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        canny = cv2.Canny(blur, 50, 150)

        # Apply the Otsu's thresholding method
        ret, otsu = cv2.threshold(canny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours in the binary image
        contours, hierarchy = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        indexReturn, approx = biggestRectangle(contours)

        # If a biggest rectangle is found, draw it on the original frame
        if indexReturn != -1:
            hull = cv2.convexHull(contours[indexReturn])
            cv2.drawContours(frame2, [hull], 0, (0, 255, 0), 3)

        cv2.imshow("Document Scanner2", frame2)

        if key == ord(" "):
                    if len(approx) >= 4:
                        selected_rect = approx.reshape(4, 2)
                        capture_mode = False
    
    else:  
            if selected_rect is not None:
                frame2 = frame.copy() 
                for point in selected_rect:
                    cv2.circle(frame2, tuple(map(int, point)), 5, (0, 0, 255), -1)            
                cv2.drawContours(frame2, [selected_rect.astype(int)], -1, (0, 255, 0), 2)
            cv2.imshow("Document Scanner2", frame2)

            if key == ord("s"):
                if selected_rect is not None:
                    warped = four_point_transform(frame, selected_rect)
                    cv2.imwrite("Img_"+ str(counter) + ".png", warped)
                    cv2.imshow("Scanned Document", warped)
                    counter += 1 

            if key == ord("r"):
                selected_rect = None
                capture_mode = True
        
    if key == ord('q'):
        break
 
# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()