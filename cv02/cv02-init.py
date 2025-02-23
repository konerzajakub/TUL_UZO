import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
plt.ion()
clear = lambda: os.system('cls')
clear()
plt.close('all')

cap = cv2.VideoCapture('cv02_hrnecek.mp4')

while True:
    ret, bgr = cap.read()
    if not ret:
        break
    #hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)
    #hist, b = np.histogram(hsv[:,:,0], 256, (0, 256))
    x1 = 100
    y1 = 100
    x2 = 200
    y2 = 200
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0))
    cv2.imshow('Image', bgr)
    key = 0xFF & cv2.waitKey(30)
    if key == 27:
        break
    
cv2.destroyAllWindows()