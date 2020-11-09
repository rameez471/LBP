import cv2, sys, os
import numpy as np
from utils import *
from sklearn import svm

def main():
    
    haar_file = '../haarcascade_frontalface_default.xml'
    datasets = 'datasets'

    (images, lables, names, id) = getDatasets(datasets)
	# Size of images
    (width, height) = (130, 100)

	# Create a Numpy array from the two lists above
    (images, labels) = [numpy.array(lis) for lis in [images, lables]]


    print('Training...')
    x = []
		
    for img in images: 
        lbp = LBP(img)
        # Concatenate the various histogram, the resulting histogram is append into feature vector
        x.append(lbp)

    model = svm.LinearSVC()
    model.fit(x, lables)

    print('Training finished...')

    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam  = cv2.VideoCapture(0)

    while True:
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:

            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))

            lbp = []
            lbp.append(LBP(face_resize))
            prediction = model.predict(lbp)[0]

            if names[prediction] != 'Unknown':
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(im,'%s - %.0f' % (names[prediction],prediction),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            else:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(im,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255))
			

        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)
        if key == 27:
            break

if __name__ == "__main__":
	main()