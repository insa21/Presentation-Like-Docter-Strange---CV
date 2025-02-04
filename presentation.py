from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import numpy as np

# Parameters
width, height = 1280, 720
gestureThreshold = 300
folderPath = "presentation"

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

# Variables
imgList = []
delay = 30
buttonPressed = False
counter = 0
drawMode = False
imgNumber = 0
delayCounter = 0
annotations = [[]]
annotationNumber = -1
annotationStart = False
hs, ws = int(height * 0.2), int(width * 0.2)  # Width and height of small image

# Load Presentation Images
if not os.path.exists(folderPath):
    print(f"Error: Folder '{folderPath}' does not exist.")
    exit()

pathImages = sorted(os.listdir(folderPath), key=len)
if not pathImages:
    print(f"Error: No images found in folder '{folderPath}'.")
    exit()

print(f"Loaded {len(pathImages)} slides from '{folderPath}'.")

prevImgNumber = imgNumber
prevCx = None  # To track hand movement direction

while True:
    # Get image frame
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture frame from webcam.")
        break
    img = cv2.flip(img, 1)

    # Read current slide
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    if imgCurrent is None:
        print(f"Error: Unable to load image '{pathFullImage}'.")
        break

    # Find the hand and its landmarks
    hands, img = detectorHand.findHands(img)  # with draw

    # Draw Gesture Threshold line
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and buttonPressed is False:  # If hand is detected
        hand = hands[0]
        cx, cy = hand["center"]
        lmList = hand["lmList"]  # List of 21 Landmark points
        fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

        # Check if all 5 fingers are up
        if fingers == [1, 1, 1, 1, 1]:  # All fingers raised
            if prevCx is None:
                prevCx = cx  # Initialize previous position
            else:
                # Detect swipe direction based on hand movement
                if cx < prevCx - 50:  # Swipe left
                    print("Swipe Left")
                    buttonPressed = True
                    if imgNumber > 0:
                        imgNumber -= 1
                        annotations = [[]]
                        annotationNumber = -1
                        annotationStart = False
                elif cx > prevCx + 50:  # Swipe right
                    print("Swipe Right")
                    buttonPressed = True
                    if imgNumber < len(pathImages) - 1:
                        imgNumber += 1
                        annotations = [[]]
                        annotationNumber = -1
                        annotationStart = False
                prevCx = cx  # Update previous position
        else:
            prevCx = None  # Reset tracking when not all fingers are up

        # Drawing logic remains unchanged
        if fingers == [0, 1, 1, 0, 0]:  # Pointer finger and middle finger up
            xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
            yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))
            indexFinger = xVal, yVal
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        if fingers == [0, 1, 0, 0, 0]:  # Only pointer finger up
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
            yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))
            indexFinger = xVal, yVal
            annotations[annotationNumber].append(indexFinger)
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
        else:
            annotationStart = False

        if fingers == [0, 1, 1, 1, 0]:  # Erase last annotation
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True

    else:
        annotationStart = False

    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False

    # Clear annotations when switching slides
    if imgNumber != prevImgNumber:
        annotations = [[]]
        annotationNumber = -1
    prevImgNumber = imgNumber

    # Draw annotations on the current slide
    for i, annotation in enumerate(annotations):
        for j in range(len(annotation)):
            if j != 0:
                cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)

    # Add webcam feed as a small overlay
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws:w] = imgSmall

    # Display slides and webcam feed
    cv2.imshow("Slides", imgCurrent)
    cv2.imshow("Image", img)

    # Keyboard input
    key = cv2.waitKey(1)
    if key == ord('q'):  # Exit
        break
    elif key == ord('s'):  # Save annotated slide
        output_path = os.path.join("output", f"annotated_slide_{imgNumber}.jpg")
        os.makedirs("output", exist_ok=True)
        cv2.imwrite(output_path, imgCurrent)
        print(f"Saved: {output_path}")

# Release resources
cap.release()
cv2.destroyAllWindows()