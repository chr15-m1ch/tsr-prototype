# penkor travay lor saaa!!
# ena sa pu fr lerla merge
# bzn train mo trsffic signe detector model avan

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
# imports
import pyttsx3 #for voice assistant

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


font = cv2.FONT_HERSHEY_SIMPLEX

# load our models
print("[INFO] loading traffic sign detector model...")
# net = load_model('trafficSignDetector.p')
detectorModel = cv2.dnn.readNet('trafficSignDetector.p' )
print("[INFO] ...traffic sign detector model loaded")

print("[INFO] loading classification model...")
classificationModel = load_model('germanSubsetModel.p')
print("[INFO] ...classification model loaded")

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

def detectTrafficSign():
    print("")

def Classification():
    print("launching video ...")
    cap = cv2.VideoCapture('videos/sample003.mp4')
    # cap = cv2.VideoCapture(0) #from webcam
    print("... done")


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

            # detect traffic signs
            # extrack ROI
            # .
            # ################### codes here
            # .
            # .

            # then process the ROI and pass it through the network
            # .
            # .
            # ..

            # process the frame
            roi = np.asarray(imgOrignal)
            roi = cv2.resize(roi, (32, 32))
            roi = preprocessing(roi)
            # cv2.imshow("Processed Image", img)
            roi = roi.reshape(1, 32, 32, 1)

            # predict frame
            predictions = classificationModel.predict(roi)
            classIndex = classificationModel.predict_classes(roi)
            probabilityValue =np.amax(predictions)

            p = probabilityValue*100

            # if prob > 0.5, display the sign and prb
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

                # #trigger voice assitant when confidence > 90 and index from last frame is not same as current frame
                # if (dummyIndex != int(classIndex) and p > 90):
                #     print("here")
                #     dummyIndex = classIndex
                #     voiceNotification(classIndex)

            cv2.imshow("Video classification", display)
        
            # if cv2.waitKey(1) and 0xFF == ord('q'):
            #     break
            c = cv2.waitKey(500) 
            if c == 27: 
                break

        frameCount = frameCount + 1

    cap.release()
    cv2.destroyAllWindows()

def detect_and_predict_tsign(frame, detectorModel, classificationModel):
    # grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the sign detections
	detectorModel.setInput(blob)
	detections = detectorModel.forward()

	# initialize our list of signs, their corresponding locations,
	# and the list of predictions from our sign mask network
	signs = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is greater than the minimum confidence (0.5 here)
		if (confidence > 0.5):
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the sign ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			sign = frame[startY:endY, startX:endX]
			sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
			sign = cv2.resize(sign, (224, 224))
			sign = img_to_array(sign)
			sign = preprocess_input(sign)

			# add the sign and bounding boxes to their respectivelists
			signs.append(sign)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one sign was detected
	if len(signs) > 0:
		# for faster inference we'll make batch predictions on *all* signs at the 
        # same time rather than one-by-one predictions in the above `for` loop
		signs = np.array(signs, dtype="float32")
		preds = classificationModel.predict(signs, batch_size=32)

	# return a 2-tuple of the sign locations and their corresponding
	# locations
	return (locs, preds)
# detect_and_predict_tsign function ends


# ############################# main part of the program ################################### #
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect signs in the frame and determine if they are wearing a
	# face mask or not
	locs, preds = detect_and_predict_tsign(frame, detectorModel, classificationModel)

	# loop over the detected sign locations and their corresponding locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output frame
		cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
