#coding:utf-8
import tensorflow as tf 
import detect_face
import dlib                     #人脸识别的库dlib
import numpy as np              #数据处理的库numpy
import cv2                      #图像处理的库OpenCv
import face_landmark
import face_pose_estimation
import os
import numpy
import blur_detector

#face detection parameters
###############################

minsize = 20 # minimum size of face
threshold = [0.6,0.7,0.7]  # three steps's threshold
factor = 0.709 # scale factor
tf.app.flags.DEFINE_string('--gpu_list', '0', '')#GPU
FLAGS = tf.app.flags.FLAGS
#facenet embedding parameters

image_size=160 #"Image size (height, width) in pixels."
pool_type='MAX' #"The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
use_lrn=False #"Enables Local Response Normalization after the first layers of the inception network."
seed=42,# "Random seed."
batch_size= None # "Number of images to process in a batch."
frame_interval=1 # frame intervals
margin = 32
#restore mtcnn model
gpu_memory_fraction=0.2

##############################



# dlib预测器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 创建cv2摄像头对象
cap = cv2.VideoCapture(0)

# cap.set(propId, value)
# 设置视频参数，propId设置的视频参数，value设置的参数值
cap.set(3, 480)

# 截图screenshoot的计数器
cnt = 0
#
def drawCube(img, imgpoints):
    
    imgpoints = np.int32(imgpoints).reshape(-1,2)

    # draw ground floor in green color
    cv2.drawContours(img, [imgpoints[:4]], -1, (0,255,0), -3)

    # draw pillars in blue color
    for i,j in zip(range(4), range(4,8)):
        cv2.line(img, tuple(imgpoints[i]), tuple(imgpoints[j]), (255,0,0), 3)

    # draw top layer in red color
    cv2.drawContours(img, [imgpoints[4:]], -1, (0,0,255), 3)

    return img




with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')


with tf.Graph().as_default():
    with tf.Session() as sess:
        while(cap.isOpened()):


            # 字体
            font = cv2.FONT_HERSHEY_SIMPLEX

            flag, im_rd = cap.read()

    # 每帧数据延时1ms，延时为0读取的是静态帧
            k = cv2.waitKey(1)

            img = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)
##################################################

            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

            rects= []
            for box in bounding_boxes:
                rec = dlib.rectangle(int(box[0]),int(box[1]),int(box[2]),int(box[3]))
                rects.append(rec)
###################################################

#########################################3
            imageVar = blur_detector.blurDetection(img)
            im_rd = cv2.putText(im_rd,"blur: "+str(round(imageVar,3)),(500,80),font,0.5,(0,0,255),1,cv2.LINE_AA)
 # ## #############################  


            shapes = face_landmark.getLandMark(img,rects)
            for shape  in shapes:
                landmark = np.matrix([[float(p.x), float(p.y)] for p in shape.parts()])
                for point in landmark:
                    
                    pos = (int(point[0, 0]), int(point[0, 1]))

                    # 利用cv2.circle给每个特征点画一个圈，共68个
                    cv2.circle(im_rd, pos, 2, color=(0, 255, 0))
                reProjectDist,eulerAngles=face_pose_estimation.getFacePose(landmark)

                #print(eulerAngles.shape)
                outEuler = eulerAngles.tolist()
                im_rd = cv2.putText(im_rd,"x: "+str(round(outEuler[0][0],3)),(500,20),font,0.5,(255,255,255),1,cv2.LINE_AA)
                im_rd = cv2.putText(im_rd,"y: "+str(round(outEuler[1][0],3)),(500,40),font,0.5,(255,255,255),1,cv2.LINE_AA)
                im_rd = cv2.putText(im_rd,"z: "+str(round(outEuler[2][0],3)),(500,60),font,0.5,(255,255,255),1,cv2.LINE_AA)
                #im_rd = drawCube(img_gray,reProjectDist)
                   
                    #cv2.putText(im_rd, str(idx + 1), pos, font, 0.2, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(im_rd, "faces: "+str(len(rects)), (20,50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

            im_rd = cv2.putText(im_rd, "s: screenshot", (20, 400), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            im_rd = cv2.putText(im_rd, "q: quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            
            # 按下s键保存
            if (k == ord('s')):
                cnt+=1
                cv2.imwrite("screenshoot"+str(cnt)+".jpg", im_rd)

            # 按下q键退出
            if(k==ord('q')):
                break

            # 窗口显示
            cv2.imshow("camera", im_rd)

        # 释放摄像头
        cap.release()

        # 删除建立的窗口
        cv2.destroyAllWindows()
                    
                            
dlib.hit_enter_to_continue()
