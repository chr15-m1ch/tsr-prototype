# better version, where model is loaded one time
# features: gui, realtime, video, image, voice notification
# working version 1, tested with subset german

from tkinter import *
from PIL import ImageTk,Image
# import runpy #run script
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import cv2
# from tensorflow import keras
from tensorflow.keras.models import load_model
# from skimage import transform
# from skimage import exposure
# from skimage import io
# from imutils import paths
import pyttsx3 

# declare my global variables
root = Tk()

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


# probability threshold and font
threshold = 0.5         
font = cv2.FONT_HERSHEY_SIMPLEX

# load the model
print("[INFO] loading model...")
model = load_model('germanSubsetModel.p')
print("[INFO] ....model loaded")



def preprocessing(img): # pre process image function
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert to grayscale
    img =cv2.equalizeHist(img)                 # apply histogram equalisition
    img = img/255                              # divide by 255
    return img

def getClassName(classNo): #used to get class name from id
    if   classNo == 0: return 'Speed Limit 60 km/h'
    elif classNo == 1: return 'Speed Limit 80 km/h'
    elif classNo == 2: return 'Give Way'
    elif classNo == 3: return 'No entry'
    elif classNo == 4: return 'Stop'
    elif classNo == 5: return 'Keep left'
    elif classNo == 6: return 'Keep Right'
    elif classNo == 7: return 'Children Crossing'

def frameClassification(s):
    # s is either a video filename or 0 (for webcam)
    cap = cv2.VideoCapture(s) 

    className = '' 
    frameCount = 0 # keep a frame count, used to skip frames
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

                className = str(getClassName(classIndex))
                print (className)
                cv2.putText(display, className, (180, 35), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

                pColor = (0,0,255) #make prob appear red by default

                if p >= 80:
                    pColor = (0,255,0) #make prob appear green if probability is greater than 80%
                
                cv2.putText(display, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, pColor, 2, cv2.LINE_AA)

                # trigger voice assitant when confidence > 90 and index from last frame is not same as current frame
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

def voiceNotification(classNo): #give voice notif from id
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

def realTime(): # function to trigger realtime btn
    # print("real time")
    # call frame classification function with 0, ie webcam
    frameClassification(0)
    
def uploadVideo(): ## function to trigger upload video btn
    # print("upload video")
    
    # select the video file
    root.filename = filedialog.askopenfilename(
        initialdir="/TSR-UI/videos", 
        title="Select a MP4 video", 
        filetypes=(("MP4 files", "*.mp4"), ("all files", "*.*")))

    # print("file location: " + root.filename)

    # get the location
    filename = root.filename
    # call frame classification function on video filename
    frameClassification(filename) 

def uploadImage(): #function to trigger upload image btn
    # print("upload image")
    # select the image file
    root.filename = filedialog.askopenfilename(
        initialdir="/TSR-UI/images", 
        title="Select an image", 
        filetypes=(("jpeg files", ".jpg"), ("png files", "*.png")))

    # print("file location: " + root.filename)
    # get the location
    filename = root.filename

    # read the image of the file chosen
    image = cv2.imread(filename)
    # cv2.imshow("Original image", image)

    imgOrignal = image

    # resize the display window
    display = cv2.resize(imgOrignal, (800, 600))

    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    # cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(display, "SIGN  CLASS:  " , (20, 35), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(display, "PROBABILITY:  ", (20, 75), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    # predict image
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue =np.amax(predictions)

    # if probabilityValue > threshold:
    #print(getClassName(classIndex))
    # cv2.putText(display,str(classIndex)+" "+str(getClassName(classIndex)), (120, 35), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(display, str(getClassName(classIndex)), (180, 35), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    p = probabilityValue*100
    pColor = (0,0,255) #make prob appear red by default

    if p >= 80:
        pColor = (0,255,0) #make prob appear green if probability is greater than 80%

    cv2.putText(display, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, pColor, 2, cv2.LINE_AA)
    cv2.imshow("Image classification", display)

def center(frm): #center a frame
    # size of the app
    app_width = 650
    app_height = 550
    # size of screen
    screen_width = frm.winfo_screenwidth()
    screen_height = frm.winfo_screenheight()
    x = (screen_width / 2) - (app_width / 2)
    y = (screen_height / 2) - (app_height / 2)
    # position window in center of screen
    frm.geometry(f'{app_width}x{app_height}+{int(x)}+{int(y)}')
    frm.mainloop()

def main():
    # first we build our user interface
    root.title('Traffic Sign Recognition using Deep Learning')
    # icon of the app
    root.iconbitmap('app_images/traffic-512.ico')
    # add image to window
    my_img = ImageTk.PhotoImage(Image.open("app_images/signs.jpg"))
    my_label = Label(image=my_img)
    my_label.pack()
    # adding btns to the window
    btnFrame = LabelFrame(root, text="Actions", padx=100, pady=10)
    btnFrame.pack(padx=10, pady=10)
    btn1 = Button(btnFrame, text="Real Time", command=realTime)
    btn1.grid(row=0, column=0, padx=5)
    btn2 = Button(btnFrame, text="Upload Video", command=uploadVideo)
    btn2.grid(row=0, column=1, padx=5)
    btn3 = Button(btnFrame, text="Upload Image", command=uploadImage)
    btn3.grid(row=0, column=2, padx=5)
    btn4 = Button(btnFrame, text="Close", command=root.quit)
    btn4.grid(row=0, column=3, padx=5)
    # center the frame
    center(root)


if __name__ == '__main__':
    main()