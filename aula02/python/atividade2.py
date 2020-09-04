#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__      = "Matheus Dib, Fabio de Miranda"


import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import sys
import auxiliar as aux

# If you want to open a video, just change v2.VideoCapture(0) from 0 to the filename, just like below
#cap = cv2.VideoCapture('hall_box_battery.mp4')

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

print("Press q to QUIT")

# Returns an image containing the borders of the image
# sigma is how far from the median we are setting the thresholds
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


mascara = input("Qual cor identificar? (MAGENTA OU CIANO): ")

while(True):
    hsv1 = None
    hsv2 = None

    if mascara == 'MAGENTA':
        hsv1, hsv2 = aux.ranges('#FF00FF')
    elif mascara == 'CIANO':
        hsv1, hsv2 = aux.ranges('#00FFFF')

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # A gaussian blur to get rid of the noise in the image
    blur = cv2.GaussianBlur(hsv,(5,5),0) 
    
    #Craindo e aplicando as máscaras
    mask = cv2.inRange(blur, hsv1, hsv2)
    
    selecao = cv2.bitwise_and(rgb, rgb, mask=mask)
    
    segmentado = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10, 10)))
    
    # Detect the edges present in the image
    bordas = auto_canny(segmentado)

    circulos = cv2.bitwise_and(rgb, rgb, mask=bordas)

    # Display the resulting frame
    cv2.imshow('Detector de circulos',circulos)
    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
