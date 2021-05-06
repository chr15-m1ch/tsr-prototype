# detect ROI using contours and passing it through the network
# i will be working on this version 


from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import pyttsx3 

# ############################## voice assistant parameters ################################### #
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
# ############################## voice assistant parameters ################################### #


font = cv2.FONT_HERSHEY_SIMPLEX

# ############################# load our classification model################################### #
print("[INFO] loading classification model...")
classificationModel = load_model('germanSubsetModel.p')
print("[INFO] ...classification model loaded")
# ############################# load our classification model################################### #


# #################################### functions ################################################# #
def preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img =cv2.equalizeHist(img)
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

def detectRoi(frame):
    crop_img = frame.copy()
    # convert frame to hsv color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lower_red = np.array([l_h, l_s, l_v])
    # upper_red = np.array([u_h, u_s, u_v])
    lower_red = np.array([0, 66, 124])
    upper_red = np.array([180, 255, 243])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)

    # perform contours detection
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.07*cv2.arcLength(cnt, True), True)
        
        x, y, w, h = cv2.boundingRect(cnt)

        # draw the bounding box
        if (area > 7000 and area <12000):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 5)
            crop_img = frame[y:y+h, x:x+w]

    # 
    try:
        cv2.imshow("cropped", crop_img)
    except NameError:
        # print("no roi detected")
        pass

    return crop_img

def Classification():
    print("launching video ...")
    # cap = cv2.VideoCapture('videos/sample003.mp4')
    cap = cv2.VideoCapture(0) #from webcam
    print("... done")

    # initialise the variables
    className = '' 
    frameCount = 0
    dummyIndex = -1

    # initialise the crop_img
    crop_img = np.zeros([100,100,3],dtype=np.uint8)
    crop_img.fill(255)

    while(cap.isOpened()):
        # read frames
        success, imgOrignal = cap.read()

        # resize the display window
        # display = cv2.resize(imgOrignal, (800, 600))
        display = imgOrignal.copy()
        
        # read every nth frame (but start with the first frame)
        if (frameCount == 0) or (frameCount%2==0) :

            # perform the detection
            # convert frame to hsv color space
            hsv = cv2.cvtColor(display, cv2.COLOR_BGR2HSV)

            # lower_red = np.array([l_h, l_s, l_v])
            # upper_red = np.array([u_h, u_s, u_v])
            lower_red = np.array([0, 66, 124])
            upper_red = np.array([180, 255, 243])

            mask = cv2.inRange(hsv, lower_red, upper_red)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel)

            # perform contours detection
            # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                approx = cv2.approxPolyDP(cnt, 0.07*cv2.arcLength(cnt, True), True)
                
                x, y, w, h = cv2.boundingRect(cnt)

                # draw the bounding box
                if (area > 7000 and area <12000):
                    # cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		            # cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    cv2.putText(display, "traffic sign", (x, y - 10 ), font, 0.75, (255, 0, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(display, (x, y), (x+w, y+h), (255, 0, 255), 2)
                    # crop the rectangle
                    crop_img = display[y:y+h, x:x+w]

            # to prevent bbox inside bbox (refering to same object)
            # Iterate contours and hierarchy:
            # for c, h in zip(contours, hier[0]):
            #     area = cv2.contourArea(c)

            #     # Check if contour has one parent and one at least on child:
            #     if (h[3] >= 0) and (h[2] >= 0):
            #         # Get the parent from the hierarchy
            #         hp = hier[0][h[3]]

            #         # Check if the parent has a parent:
            #         if hp[3] >= 0:
            #             # Get bounding rectange
            #             x, y, w, h = cv2.boundingRect(c)
                    
            #         if (area > 7000 and area <12000):

            #             # Draw red rectange for testing
            #             cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), thickness=1)



            # 
            # try:
            #     cv2.imshow("cropped", crop_img)
            # except NameError:
            #     # print("no roi detected")
            #     pass
            # # detection
            try:
                cv2.imshow("cropped", crop_img)
            except NameError:
                print("no roi detected")
                pass
            # # detection

            # # detect traffic signs (get the ROI)
            # img_crop = detectRoi(imgOrignal)
            # roi = img_crop.copy()
            roi = crop_img.copy()
            try:
                # cv2.imshow("cropped", crop_img)
                # process the ROI for classification
                roi = np.asarray(roi)
                roi = cv2.resize(roi, (32, 32))
                roi = preprocessing(roi)
                # cv2.imshow("Processed Image", img)
                roi = roi.reshape(1, 32, 32, 1)

                # predict ROI
                predictions = classificationModel.predict(roi)
                classIndex = classificationModel.predict_classes(roi)
                probabilityValue =np.amax(predictions)

                p = probabilityValue*100

                # if prob > 0.5, display the clas sign and prb
                if probabilityValue > 0.5:

                    cv2.putText(display, "SIGN CLASS: " , (20, 35), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(display, "PROBABILITY: ", (20, 75), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

                    className = str(getClassName(classIndex))
                    print (className)
                    cv2.putText(display, className, (180, 35), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

                    pColor = (0,0,255) #make prob appear red by default

                    if p >= 80:
                        pColor = (0,255,0) #make prob appear green if probability is greater than 80%

                    cv2.putText(display, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, pColor, 2, cv2.LINE_AA)

                    #trigger voice assitant when confidence > 90 and index from last frame is not same as current frame
                    if (dummyIndex != int(classIndex) and p > 90):
                        # print("here")
                        dummyIndex = classIndex
                        engine.say(className)
                        engine.runAndWait()
                        engine.stop()
            
            except NameError:
                # print("no roi detected")
                pass


            cv2.imshow("Video classification", display)
            cv2.imshow("cropped", crop_img)

        
            # if cv2.waitKey(1) and 0xFF == ord('q'):
            #     break
            c = cv2.waitKey(500) 
            if c == 27: 
                break

        frameCount = frameCount + 1

    cap.release()
    cv2.destroyAllWindows()

# #################################### functions ################################################# #



# ############################# main part of the program ################################### #
# calling the main function
Classification()
