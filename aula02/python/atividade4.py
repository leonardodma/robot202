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


def encontra_circulo(img, codigo_cor):
    # MAGENTA
    hsv_1, hsv_2 = aux.ranges(codigo_cor)

    # convert the image to grayscale, blur it, and detect edges
    hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)

    #blur = cv2.GaussianBlur(hsv, (5, 5), 0)

    mask = cv2.inRange(hsv, hsv_1, hsv_2)

    #res = cv2.bitwise_and(hsv, hsv, mask=mask_m)
    #res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

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

    circles_magenta = encontra_circulo(frame, '#FF00FF')
    circles_ciano = encontra_circulo(frame, '#7efcfc')


    a_m, b_m, r_m = None, None, None
    a_c, b_c, r_c = None, None, None


    if circles_magenta is not None:
        circles_magenta = np.uint16(np.around(circles_magenta)) 
    
        # Desenha círculos da cor Magenta 
        for pt in circles_magenta[0, :]: 
            a_m, b_m, r_m = pt[0], pt[1], pt[2] 
    
            # Draw the circunference of the circle. 
            cv2.circle(frame, (a_m, b_m), r_m, (0, 255, 255), 2) 
    
            # Draw a small circle (of radius 1) to show the center. 
            cv2.circle(frame, (a_m, b_m), 1, (0, 255, 255), 3)


    if circles_ciano is not None: 
        circles_ciano= np.uint16(np.around(circles_ciano)) 
        # Desenha círculos da cor ciano 
        for pt in circles_ciano[0, :]: 
            a_c, b_c, r_c = pt[0], pt[1], pt[2] 
    
            # Draw the circunference of the circle. 
            cv2.circle(frame, (a_c, b_c), r_c, (0, 255, 255), 2) 
    
            # Draw a small circle (of radius 1) to show the center. 
            cv2.circle(frame, (a_c, b_c), 1, (0, 255, 255), 3)


    #cv2.line(frame, tuple[a_m, b_m], tuple(a_c, b_c), (0, 255, 0), thickness=3, lineType=8)

    cv2.imshow("Detected Circle", frame)

    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()