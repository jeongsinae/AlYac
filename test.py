import numpy as np
import cv2
import matplotlib.pyplot as plt

def imshow(img):
	# cv2.imshow('test', img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	plt.imshow(img)
	plt.show()

def contour():
	img = cv2.imread('./image/test2.png')
	imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	ret, thr = cv2.threshold(imgray, 127, 255, 0)
	_, contours = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	imshow(imgray)
	imshow(thr)
	cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
	imshow(thr)
	imshow(img)
