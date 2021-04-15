# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
from pygame import mixer
# import time
# time.sleep(3)

mixer.init()
sound = mixer.Sound('alarm.wav')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')
lbl=['Close','Open']
sound2= mixer.Sound("yawn_sound.wav")
model = load_model('models/cnn_drowsiness.h5')
# cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter('output_drowsiness.avi',fourcc, 5, (640,480))

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(ht, wt) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	# print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([wt, ht, wt, ht])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(wt - 1, endX), min(ht - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("./models/yawn_new_test_1.model")

# initialize the video stream
# print("[INFO] starting video stream...")
cap=cv2.VideoCapture(-1)
cap.open(0, cv2.CAP_DSHOW)

# loop over the frames from the video stream
core_yawn=0
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	ret,frame = cap.read()
	frame = imutils.resize(frame, width=640)
	frame= cv2.flip(frame,1)
	height,width = frame.shape[:2]
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	left_eye = leye.detectMultiScale(gray)
	right_eye =  reye.detectMultiScale(gray)
	for (x,y,w,h) in right_eye:
		r_eye=frame[y:y+h,x:x+w]
		count=count+1
		r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
		r_eye = cv2.resize(r_eye,(24,24))
		r_eye= r_eye/255
		r_eye=  r_eye.reshape(24,24,-1)
		r_eye = np.expand_dims(r_eye,axis=0)
		rpred = model.predict_classes(r_eye)
		if(rpred[0]==1):
			lbl='Open' 
			if(rpred[0]==0):
				lbl='Closed'
				break
	for (x,y,w,h) in left_eye:
		l_eye=frame[y:y+h,x:x+w]
		count=count+1
		l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
		l_eye = cv2.resize(l_eye,(24,24))
		l_eye= l_eye/255
		l_eye=l_eye.reshape(24,24,-1)
		l_eye = np.expand_dims(l_eye,axis=0)
		lpred = model.predict_classes(l_eye)
		if(lpred[0]==1):
			lbl='Open'   
		if(lpred[0]==0):
			lbl='Closed'
		break

	if(rpred[0]==0 and lpred[0]==0):
		score=score+1
		cv2.putText(frame,"Closed",(10,height-20), font, 1,(0,0,255),1,cv2.LINE_AA)

	# if(rpred[0]==1 or lpred[0]==1):
	else:
		score=score-1
		cv2.putText(frame,"Open",(10,height-20), font, 1,(0,255,0),1,cv2.LINE_AA)

	if(score<0):
		score=0
	cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
	if(score>=8):
		try:
			sound.play()

		except:  # isplaying = False
			pass
		if(thicc<16):
			thicc= thicc+2
		else:
			thicc=thicc-2
			if(thicc<2):
				thicc=2
		cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(no_yawn, yawn) = pred

		if no_yawn > yawn:
			label = "no_yawn"
		elif no_yawn<= yawn:
			label= "yawn"
			core_yawn+=1
		if label == "no_yawn":
			color=(0,255,0)
			core_yawn=0
		elif label=="yawn":
			color=(0, 0, 255)

		if core_yawn==3:
			try:
				sound2.play()

			except:  # isplaying = False
				pass		

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)
	# show the output frame

	cv2.imshow("MY_PROGRAM", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break


cv2.release()

cv2.destroyAllWindows()