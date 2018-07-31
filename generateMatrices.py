import cv2
import math


# trying to speed up convergence by having input image arrays that highlight features
# not sure yet if this will be super useful
def generateMatrices(image):
	channels = cv2.split(image) 
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# color quantization
	array = gray.reshape((-1,3))
	array = np.float32(array)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.3)
	clusters = 10
	compactness,label,center=cv2.kmeans(gray,clusters,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	center = np.uint8(center)
	centers = center[label.flatten()]
	kClustered = centers.reshape((gray.shape))

	blur = cv2.GaussianBlur(gray,(5,5),0)
	sharpen = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

	laplacian = cv2.Laplacian(image,cv2.CV_64F)
	laplacianGray = cv2.cvtColor(laplacian, cv2.COLOR_BGR2GRAY)


	return [channels[0], channels[1], channels[2], gray, kClustered, blur, sharpen, laplacianGray]
