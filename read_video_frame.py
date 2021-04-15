import cv2 

# cap = cv2.VideoCapture('videos/sample003.mp4')
cap = cv2.VideoCapture(0)

count = 0
while(True):
    ret, frame = cap.read()

    if count%10==0:
        cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count = count+1
cv2.destroyAllWindows()