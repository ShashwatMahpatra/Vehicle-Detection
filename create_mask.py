import cv2
import numpy as np

def MakeMask(coordinates,image_size):

	#Creating the frame
	image = np.zeros(image_size, dtype = np.uint8)
	points = np.array(coordinates, np.int32)
	points = points.reshape((-1,1,2))
	#Making the mask
	cv2.fillPoly(image,[points],(255,255,255))
	#cv2.polylines(image, [points] , True, (255,255,255),-1)
	#image = cv2.flip(image,0)
	#cv2.imwrite("mask_jumppoint.jpg",image)
	#cv2.imshow("temp",image)
	cv2.waitKey()
	cv2.destroyAllWindows()
	#Returning the mask
	return image
	
if __name__ == "__main__":
	MakeMask([[377,310],[353,480], [615,480], [555,310]],(576,720,3))