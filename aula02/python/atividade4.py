# Referências:
# https://www.geeksforgeeks.org/circle-detection-using-opencv-python/   
# https://stackoverflow.com/questions/47349833/draw-line-between-two-given-points-opencv-python


import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import sys
import auxiliar as aux
import imutils
from imutils import paths   


def encontra_marcador(img):
    # MAGENTA
    hsv_m1, hsv_m2 = aux.ranges('#FF00FF')
    # CIANO
    hsv_c1, hsv_c2 = aux.ranges('#00FFFF')


    # convert the image to grayscale, blur it, and detect edges
    hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask_m = cv2.inRange(hsv, hsv_m1, hsv_m2)
    mask_c = cv2.inRange(hsv, hsv_c1, hsv_c2)
    mask = mask_c + mask_m

    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    edged = cv2.Canny(blur, 35, 125)


    circles=cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=5,maxRadius=100)
    
    return circles


if len(sys.argv) > 1:
    arg = sys.argv[1]
    try:
        input_source=int(arg) # se for um device
    except:
        input_source=str(arg) # se for nome de arquivo
else:   
    input_source = 0

cap = cv2.VideoCapture(input_source)

# Parameters to use when opening the webcam.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

lower = 0
upper = 1


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    circles = encontra_marcador(frame)


    if circles is not None: 
  
    # Convert the circle parameters a, b and r to integers. 
        circles = np.uint16(np.around(circles)) 
    
        for pt in circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 
    
            # Draw the circumference of the circle. 
            cv2.circle(frame, (a, b), r, (0, 255, 0), 2) 
    
            # Draw a small circle (of radius 1) to show the center. 
            cv2.circle(frame, (a, b), 1, (0, 0, 255), 3) 
            

    cv2.imshow("Detected Circle", frame) 

    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()