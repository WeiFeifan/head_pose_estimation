import numpy as np 
import cv2

def blurDetection(img):
	imageVar = cv2.Laplacian(img,cv2.CV_64F).var()
	return imageVar
