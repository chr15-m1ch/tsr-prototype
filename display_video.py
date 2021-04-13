# imports
import numpy as np
import cv2
import pickle
from tensorflow import keras
from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from skimage import io
from imutils import paths
import argparse
import imutils
import random
import os
import argparse
from tkinter import *
from PIL import ImageTk,Image
import runpy
from tkinter import filedialog

# global videoFile
root = Tk()
root.withdraw()

# app_width = 10
# app_height = 10
# # size of screen
# screen_width = root.winfo_screenwidth()
# screen_height = root.winfo_screenheight()

# x = (screen_width / 2) - (app_width / 2)
# y = (screen_height / 2) - (app_height / 2)

# # position window in center of screen
# root.geometry(f'{app_width}x{app_height}+{int(x)}+{int(y)}')
# root.mainloop()

# print("upload video")
# runpy.run_path("prog.py")

root.filename = filedialog.askopenfilename(
    initialdir="/TSR-UI/images", 
    title="Select a MP4 video", 
    filetypes=(("MP4 files", "*.mp4"), ("all files", "*.*")))

# print("file location: " + root.filename)
filename = root.filename
# videoFile = filename
# script = "display_video.py --folder_path=" + filename

# print(script)
# runpy.run_path("display_video.py --folder_path=C:/Users/SURJU/Desktop/tsr-prototype-with-gui/videos/sample001.mp4")


# ap = argparse.ArgumentParser()
# ap.add_argument('--folder_path',type=str,help='Specify folder path (OPTIONAL)')

# # args = vars(ap.parse_args())
# args = ap.parse_args()


threshold = 0.80         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

# load the model
print("[INFO] loading model...")
model = load_model('germanSubsetModel.p')

 
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

# calling function 

# read the video
cap = cv2.VideoCapture(filename)

while(cap.isOpened()):
    # READ IMAGE
    success, imgOrignal = cap.read()

    display = cv2.resize(imgOrignal, (800, 600))
    
    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    # cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(display, "CLASS: " , (20, 35), font, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(display, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue =np.amax(predictions)

    if probabilityValue > threshold:
        #print(getClassName(classIndex))
        cv2.putText(display,str(classIndex)+" "+str(getClassName(classIndex)), (120, 35), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

        p = probabilityValue*100
        pColor = (0,0,255) #make prob appear red by default

        if p >= 80:
            pColor = (0,255,0) #make prob appear green if probability is greater than 80%

        cv2.putText(display, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, pColor, 2, cv2.LINE_AA)
        cv2.imshow("Video classification", display)
    
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()