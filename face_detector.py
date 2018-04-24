import tensorflow as tf
import cv2
import os
import time
import detect_face
import dlib
#face detection parameters
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

def getFace(img):
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')

    with tf.Graph().as_default():
        with tf.Session() as sess:
            bounding_boxes, _  = detect_face.detect_face(img,minsize,pnet,rnet,onet,threshold,factor)

    
    recs= []
    for box in bounding_boxes:
        #print(bounding_boxes[i][0:4])
        rec = dlib.rectangle(int(box[0]),int(box[1]),int(box[2]),int(box[3]))
        recs.append(rec)          
    return recs
    
"""
def getFace(path):
    res = []
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')

    with tf.Graph().as_default():
        with tf.Session() as sess:
            for root,dirfiles,filenames in os.walk(dirpath):
                for x in filenames:
                    img_path = os.path.join(root,x)
                    print(img_path)
                    # with tf.device("/gpu:0"):
                    frame = cv2.imread(img_path)
                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]  # number of faces
            
    #return bounding_boxes
    """

if __name__ =="__main__":

    dirpath = '/home/shaohe/wff/face_detect/test_data/rec_result_20180409/caoxiaoyi'
    start_time = time.time()
    for root,dirfiles,filenames in os.walk(dirpath):
          for x in filenames:
                img_path = os.path.join(root,x)
                print(img_path)
                frame = cv2.imread(img_path)
                #getFace(frame)
                print(getFace(frame).shape[0])

    print(time.time()-start_time)