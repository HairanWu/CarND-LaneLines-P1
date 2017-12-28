import cv2
import numpy as np
import sys


InputFileName = "test_images/challenge1.jpg"

WindowName = "SelectColorDemo V1.0 Created by WHR"

img = cv2.imread(InputFileName)

try:
	cv2.namedWindow(WindowName)
	cv2.imshow(WindowName,img)
except Exception as err:   
	print ('Exception: ', err)
	print("Open file %s failed!"%(InputFileName))
	sys.exit(-1)
	
def nothing(x):
	pass
	
DevCH1LowerName = 'Ch1L'
DevCH1UpperName = 'Ch1U'
DevCH2LowerName = 'Ch2L'
DevCH2UpperName = 'Ch2U'
DevCH3LowerName = 'Ch3L'
DevCH3UpperName = 'Ch3U'

cv2.createTrackbar(DevCH1LowerName,WindowName,0,255,nothing)
cv2.createTrackbar(DevCH1UpperName,WindowName,0,255,nothing)

cv2.createTrackbar(DevCH2LowerName,WindowName,0,255,nothing)
cv2.createTrackbar(DevCH2UpperName,WindowName,0,255,nothing)

cv2.createTrackbar(DevCH3LowerName,WindowName,0,255,nothing)
cv2.createTrackbar(DevCH3UpperName,WindowName,0,255,nothing)


cv2.setTrackbarPos(DevCH1LowerName,WindowName,0)
cv2.setTrackbarPos(DevCH1UpperName,WindowName,255)

cv2.setTrackbarPos(DevCH2LowerName,WindowName,0)
cv2.setTrackbarPos(DevCH2UpperName,WindowName,255)

cv2.setTrackbarPos(DevCH3LowerName,WindowName,0)
cv2.setTrackbarPos(DevCH3UpperName,WindowName,255)

switch = '0 : RGB \n1 : HSV \n2 : HSL'
cv2.createTrackbar(switch,WindowName,0,2,nothing)


def colorselect(inputimg,ch1_l,ch2_l,ch3_l,ch1_u,ch2_u,ch3_u):
	if ch1_l<ch1_u and ch2_l<ch2_u and ch3_l<ch3_u:
		mode = cv2.getTrackbarPos(switch,WindowName)
		if mode == 0:
			colormode = cv2.COLOR_BGR2RGB
		elif mode == 1:
			colormode = cv2.COLOR_BGR2HSV
		elif mode == 2:
			colormode = cv2.COLOR_BGR2HLS

		converted = cv2.cvtColor(inputimg, colormode)
		lower_color = np.array([ch1_l, ch2_l, ch3_l]) 
		upper_color = np.array([ch1_u, ch2_u, ch3_u]) 
		color_mask = cv2.inRange(converted, lower_color, upper_color)
	
		mask = color_mask

		dst = cv2.bitwise_and(inputimg, inputimg, mask = mask)
		return dst
	else:
		return inputimg

if __name__ == '__main__':
	while(1):
		ch1l = cv2.getTrackbarPos(DevCH1LowerName,WindowName)
		ch1u = cv2.getTrackbarPos(DevCH1UpperName,WindowName)
		ch2l = cv2.getTrackbarPos(DevCH2LowerName,WindowName)
		ch2u = cv2.getTrackbarPos(DevCH2UpperName,WindowName)
		ch3l = cv2.getTrackbarPos(DevCH3LowerName,WindowName)
		ch3u = cv2.getTrackbarPos(DevCH3UpperName,WindowName)

		filtedimg = colorselect(img,ch1l,ch2l,ch3l,ch1u,ch2u,ch3u)

		cv2.imshow(WindowName,filtedimg)
		
		#ESC
		k = cv2.waitKey(10) & 0xFF

		if k == 27:
			break
	
	cv2.destroyAllWindows()
