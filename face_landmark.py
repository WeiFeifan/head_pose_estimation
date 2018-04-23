import dlib
import cv2
import face_detector
import numpy
predictor_path = "shape_predictor_68_face_landmarks.dat"

predictor = dlib.shape_predictor(predictor_path)

def getLandMark(img,recs):
	shapes = []
	for rec in recs:
		shape=predictor(img,rec)
		shapes.append(shape)
	return shapes

if __name__ =="__main__":
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
	shapes = getLandMark(img,ds)
	for shape  in shapes:
		landmark = numpy.matrix([[p.x, p.y] for p in shape.parts()])
		print("face_landmark:")
		print (landmark) 
		
		win.add_overlay(shape)
	dlib.hit_enter_to_continue()
