import cv2,sys,numpy,os,time

def localBinaryPattern_wh(images):
    (w,h,c) = images.shape
    images_processed = numpy.zeros(())