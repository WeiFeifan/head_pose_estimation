import numpy as np 
import cv2
import face_landmark
import face_detector
import dlib
camMat = np.array([[6.5308391993466671e+002, 0.0, 3.1950000000000000e+002],[0.0, 6.5308391993466671e+002, 2.3950000000000000e+002],[0.0, 0.0, 1.0]]) 
distCoffs = np.array([ 7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000])
import numpy


objPts = np.float32([[6.825897, 6.760612, 4.402142],
	[1.330353, 7.122144, 6.903745],
	[-1.330353, 7.122144, 6.903745],
	[-6.825897, 6.760612, 4.402142],
	[5.311432, 5.485328, 3.987654],
	[1.789930, 5.393625, 4.413414],
	[-1.789930, 5.393625, 4.413414],
	[-5.311432, 5.485328, 3.987654],
	[2.005628, 1.409845, 6.165652],
	[-2.005628, 1.409845, 6.165652],
	[2.774015, -2.080775, 5.048531],
	[-2.774015, -2.080775, 5.048531],
	[0.000000, -3.116408, 6.097667],
	[0.000000, -7.415691, 4.070434]])



reProjectSrc = np.float32([[10.0, 10.0, 10.0], [10.0, 10.0, -10.0], [10.0, -10.0, -10.0],[10.0, -10.0, 10.0],
 [-10.0, 10.0, 10.0],[-10.0, 10.0, -10.0], [-10.0, -10.0, -10.0], [-10.0, -10.0, 10.0]])



def getImagePts(imagePts):
	res = np.concatenate((imagePts[17],imagePts[21],imagePts[22],imagePts[26],imagePts[36],imagePts[39],
		imagePts[42],imagePts[45],imagePts[31],imagePts[35],imagePts[48],imagePts[54],imagePts[57],imagePts[8]))
	return res

def getFacePose(imagePts):
	
	imagePts = getImagePts(imagePts)
	_,rotVects, transVects = cv2.solvePnP(objPts, imagePts, camMat, distCoffs)

	reProjectDist, jac = cv2.projectPoints(reProjectSrc, rotVects, transVects, camMat, distCoffs)

	#print(rotVects.shape)
	rotationMat,_ = cv2.Rodrigues(rotVects)
	#print(transVects.shape)
	poseMat  = np.column_stack((rotationMat,transVects))
	_,_,_,_,_,_,eulerAngles = cv2.decomposeProjectionMatrix(poseMat,camMat,rotVects,transVects)

	return reProjectDist, eulerAngles
if __name__ == "__main__":
	faces_path = "/home/shaohe/wff/face_detect/test_data/rec_result_20180409/caoxiaoyi/20180409124325.jpg"
	'''加载人脸检测器、加载官方提供的模型构建特征提取器'''
	detector = dlib.get_frontal_face_detector()
	win = dlib.image_window()
	#imgbgr = io.imread(faces_path)
	imgbgr = cv2.imread(faces_path)
	img = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB) 

	print("img:")
	print(img.shape[::-1][1:])
	win.clear_overlay()
	win.set_image(img)

	ds = face_detector.getFace(img)

	print(type(ds[0]))
	shapes = face_landmark.getLandMark(img,ds)
	for shape  in shapes:
		landmark = numpy.matrix([[float(p.x), float(p.y)] for p in shape.parts()])
		

		print("face_landmark:")
		print (type(landmark))
		print(landmark)
		print("++++++++++++++")
		reProjectDist,eulerAngles=getFacePose(landmark)
		print("reProjectDist:")
		print(reProjectDist) 
		print("eulerAngles:")
		print(eulerAngles)
		win.add_overlay(shape)
	dlib.hit_enter_to_continue()
