import cv2
import numpy as np
import sys

# Command arguments
videoFile = sys.argv[1]
# imgFile = sys.argv[2]
background = sys.argv[2]
outputFile = sys.argv[3]

# Inputs
video = cv2.VideoCapture(videoFile)
# image = cv2.imread(imgFile)
bg = cv2.VideoCapture(background)

# video caracteristics
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)  
nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# recorder to save the output
recorder = cv2.VideoWriter(outputFile, 
                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                                fps, 
                                (640, 480))

while True:
    ret, frame = video.read()
    ret2, frame2 = bg.read()
    frame = cv2.resize(frame, (640, 480))
    # image = cv2.resize(image, (640, 480))
    frame2 = cv2.resize(frame2, (640, 480))

    # For each frame
    # Change color to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    g = 75 # Hue value for the green
    margin = 2 # For the lower and upper values of green
    mask = cv2.inRange(h, g - margin, g + margin) # Mask to select the green values (green screen)

    dilatation = np.ones((3, 3), np.uint8)  
    mask = cv2.dilate(mask, dilatation, iterations=1) # Dilatation of the mask to select more values of green that are located 
                                                      # in the boundaries and are not in the range of the initial mask
    mask = cv2.medianBlur(mask, 5) # Make pixels values more homogeneous
    mask = cv2.bitwise_not(mask)  # Inverse the mask (bit 0 for green screen)

    foreground = np.zeros_like(frame)  
    foreground[mask == 255] = frame[mask == 255]  # Apply the mask to the original video

    result = np.where(foreground == 0, frame2, foreground) # Adding the background (replace zeros by the background)

    # Show live videos 
    cv2.imshow("mask", mask)
    cv2.imshow("foreground", foreground)
    cv2.imshow("result", result)

    recorder.write(result)

    if cv2.waitKey(25) == 27:
        break

recorder.release()
video.release()
cv2.destroyAllWindows()
