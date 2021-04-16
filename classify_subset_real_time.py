# imports
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from skimage import io
from imutils import paths
 
frameWidth= 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.80         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
 
# capture frames from web cam
cap = cv2.VideoCapture(0)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)
# cap.set(10, brightness)

# load our trained model
print("[INFO] loading model...")
model = load_model('germanSubsetModel.p')

# preprocessing 
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img =cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

def getClassName(classNo):
    if   classNo == 0: return 'Speed Limit 60 km/h'
    elif classNo == 1: return 'Speed Limit 80 km/h'
    elif classNo == 2: return 'Give Way'
    elif classNo == 3: return 'No entry'
    elif classNo == 4: return 'Stop'
    elif classNo == 5: return 'Keep left'
    elif classNo == 6: return 'Keep Right'
    elif classNo == 7: return 'Children Crossing'


while True:
    # read frame
    success, imgOriginal = cap.read()
    display = cv2.resize(imgOriginal, (32, 32))
    # pre process image
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    # cv2.imshow("Processed Image", img)

    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOriginal, "CLASS: " , (20, 35), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    # predict image
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue =np.amax(predictions)

    if probabilityValue > threshold:
        #print(getClassName(classIndex))
        # cv2.putText(imgOriginal,str(classIndex)+" "+str(getClassName(classIndex)), (120, 35), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display, str(getClassName(classIndex)), (180, 35), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

        p = probabilityValue*100
        pColor = (0,0,255) #make prob appear red by default

        if p >= 80:
            pColor = (0,255,0) #make prob appear green if probability is greater than 80%

        cv2.putText(imgOriginal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, pColor, 2, cv2.LINE_AA)
        cv2.imshow("Real Time Classification", imgOriginal)
    
        # if cv2.waitKey(1) and 0xFF == ord('q'):
        #     break

        # get value pressed
        # press esc to exit window
        c = cv2.waitKey(500) 
        if c == 27: 
            break
