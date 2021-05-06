import cv2
import numpy as np

# img = cv2.imread('images/ROI images/no entry.jpg')
img = np.zeros([800,600,3],dtype=np.uint8)
img.fill(255)
display = cv2.resize(img, (800, 600))

cv2.putText(display, "CLASS: " , (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(display, "CLASS: " , (20, 75), cv2.FONT_HERSHEY_PLAIN , 0.75, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(display, "CLASS: " , (20, 115), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(display, "CLASS: " , (20, 155), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(display, "CLASS: " , (20, 195), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(display, "CLASS: " , (20, 235), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.75, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(display, "CLASS: " , (20,275), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX , 0.75, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(display, "CLASS: " , (20, 315), cv2.FONT_HERSHEY_SCRIPT_COMPLEX  , 0.75, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow('image', display)
cv2.waitKey(0)
