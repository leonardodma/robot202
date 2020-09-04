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

mascara = input("Qual a cor do circulo que você quer identificar? (MAGENTA OU CIANO): ")

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
    
    segmentado = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((4, 4)))

    selecao = cv2.bitwise_and(rgb, rgb, mask=segmentado)
    
    contornos, arvore = cv2.findContours(segmentado.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contornos_img = rgb.copy()
    cv2.drawContours(contornos_img, contornos, -1, [0, 0, 255], 3);

    # Display th e resulting frame
    cv2.imshow('Detector de circulos', contornos_img)
    
    # cv2.imshow('Detector de circulos', rgb)
    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()