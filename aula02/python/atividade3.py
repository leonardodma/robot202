# Referências:
# https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/


import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import sys
import auxiliar as aux
import imutils
from imutils import paths   


def encontra_marcador(img):
        # convert the image to grayscale, blur it, and detect edges
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 35, 125)
        # find the contours in the edged image and keep the largest one;

        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key = cv2.contourArea)
        # compute the bounding box of the of the paper region and return it
        return cv2.minAreaRect(c)


def encontra_distancia(H, h, f):
    distance = (H*f)/h
	
    return distance


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


# Calibration - More details in aula2/Atividade2

KD = 30 # Know distance (cm)
H = 7.6 # Real distance between the two circle centers (cm)
h = 60.5 # Distânce beyween the two circle centers in the screen  (pixel) 
f = (h*KD)/H #Pixel


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()


    # loop over the images
    for imagePath in sorted(paths.list_images(frame)):
        # load the image, find the marker in the image, then compute the
        # distance to the marker from the camera
        image = cv2.imread(imagePath)
        marker = encontra_marcador(image)
        inches = encontra_distancia(H, h, marker[1][0])
        # draw a bounding box around the image and display it
        box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
        box = np.int0(box)
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
        cv2.putText(image, "%.2fft" % (inches / 12),
            (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
            2.0, (0, 255, 0), 3)
        cv2.imshow(frame)
    

    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()