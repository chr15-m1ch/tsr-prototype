# penkor travay lor saaa!!
# ena sa pu fr lerla merge
# kav ajoute la voix ladn

# imports
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from skimage import io
from imutils import paths
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import pyttsx3

# voice assistant parameters
engine = pyttsx3.init() # object creation
# RATE
rate = engine.getProperty('rate')   # getting details of current speaking rate
engine.setProperty('rate', 200)     # setting up new voice rate
# VOLUME
volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
engine.setProperty('volume', 1.0)    # setting up volume level  between 0 and 1
# VOICE
voices = engine.getProperty('voices')       #getting details of current voice
engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
# engine.say("Speed Limit 60 km/h")
# engine.runAndWait()
# engine.stop()

# GUI
root = Tk()
root.withdraw() #destroy the root window

# select the video file
# root.filename = filedialog.askopenfilename(
#     initialdir="/TSR-UI/images", 
#     title="Select a MP4 video", 
#     filetypes=(("MP4 files", "*.mp4"), ("all files", "*.*")))

# # print("file location: " + root.filename)
# # get the location
# filename = root.filename

# probability threshold
threshold = 0.5         
font = cv2.FONT_HERSHEY_SIMPLEX

# load the model
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

def voiceNotification(classNo):
    if   classNo == 0: 
        engine.say("Speed Limit 60 km/h")
        engine.runAndWait()
        engine.stop()

    elif classNo == 1: 
        engine.say("Speed Limit 80 km/h")
        engine.runAndWait()
        engine.stop()

    elif classNo == 2: 
        engine.say("Give Way")
        engine.runAndWait()
        engine.stop()

    elif classNo == 3: 
        engine.say("No entry")
        engine.runAndWait()
        engine.stop()

    elif classNo == 4: 
        engine.say("Stop")
        engine.runAndWait()
        engine.stop()

    elif classNo == 5: 
        engine.say("Keep left")
        engine.runAndWait()
        engine.stop()

    elif classNo == 6: 
        engine.say("Keep Right")
        engine.runAndWait()
        engine.stop()

    elif classNo == 7: 
        engine.say("Children Crossing")
        engine.runAndWait()
        engine.stop()


# read the video of the file chosen
# cap = cv2.VideoCapture(filename) # use when choosing file
# cap = cv2.VideoCapture('videos/sample001.mp4')

cap = cv2.VideoCapture(0) #from webcam

className = '' 
frameCount = 0
dummyIndex = -1

while(cap.isOpened()):
    # read frames
    success, imgOrignal = cap.read()

    # resize the display window
    display = cv2.resize(imgOrignal, (800, 600))
    
    # pre process the nth frame (but start with the first frame)
    if (frameCount == 0) or (frameCount%10==0) :
        # cv2.imshow('frame',frame)
        # process the frame
        img = np.asarray(imgOrignal)
        img = cv2.resize(img, (32, 32))
        img = preprocessing(img)
        # cv2.imshow("Processed Image", img)
        img = img.reshape(1, 32, 32, 1)

        # predict frame
        predictions = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue =np.amax(predictions)

        p = probabilityValue*100

        # if prob > 0.5, display the sign and prb
        if probabilityValue > threshold:

            cv2.putText(display, "SIGN CLASS: " , (20, 35), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display, "PROBABILITY: ", (20, 75), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)


            # if probabilityValue > threshold:
                #print(getClassName(classIndex))
            className = str(getClassName(classIndex))
            print (className)
            cv2.putText(display, className, (180, 35), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

            pColor = (0,0,255) #make prob appear red by default

            if p >= 80:
                pColor = (0,255,0) #make prob appear green if probability is greater than 80%

            cv2.putText(display, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, pColor, 2, cv2.LINE_AA)
        
            # print(int(classIndex))
            # print(dummyIndex)
            # print(dummyIndex != int(classIndex))
            # print(dummyIndex != -1)

            if (dummyIndex != int(classIndex) and p > 90):
                print("here")
                dummyIndex = classIndex
                voiceNotification(classIndex)
            # voice assistant

        cv2.imshow("Video classification", display)
    
        # if cv2.waitKey(1) and 0xFF == ord('q'):
        #     break
        c = cv2.waitKey(500) 
        if c == 27: 
            break

    frameCount = frameCount + 1

cap.release()
cv2.destroyAllWindows()