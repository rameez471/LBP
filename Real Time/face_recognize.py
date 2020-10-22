import cv2, sys, os
import numpy as np
from utils import *

def main():
    
    haar_file = '../haarcascade_frontalface_default.xml'
    datasets = 'datasets'

    (images, lables, names, id) = getDatasets(datasets)
	# Size of images
    (width, height) = (130, 100)

	# Create a Numpy array from the two lists above
    (images, lables) = [numpy.array(lis) for lis in [images, lables]]

    print('Training...')

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(images,lables)

    print('Training finished...')

    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam  = cv2.VideoCapture(0)

    while True:
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))

			# if arguments.algorithm == "fisherface":
			# 	# Try to recognize the face using fisherface
			# 	prediction = model.predict(face_resize) #fisherface get the label as 0
			# 	#print("[Debug]: " + names[prediction] + "'s face found")
			# if arguments.algorithm == "lbp":
			# 	lbp = [];
			# 	lbp.append(LBP(face_resize))
			# 	prediction = model.predict(lbp)
			# 	print(prediction)
			# 	prediction = prediction[0] # SVM get and array with the class [0]
			# 	# print("[Debug]: " + names[prediction] + "'s face found")

            prediction,confidence = face_recognizer.predict(face_resize)

            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

            if confidence < 80:
                cv2.putText(im,'%s - %.0f' % (names[prediction],prediction),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            else:
                cv2.putText(im,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                print('Not recognized')
                print(confidence)
			

        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)
        if key == 27:
            break

if __name__ == "__main__":
	main()