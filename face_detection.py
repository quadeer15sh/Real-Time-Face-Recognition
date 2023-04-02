import cv2  
import os
import time
from threading import Thread
import numpy as np 
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances as L2
import math
import mediapipe as mp
from mtcnn import MTCNN
import scipy.stats as st
import warnings

warnings.filterwarnings("ignore")

class FaceDetection:

    def __init__(self,model,path,mode,thresh=1.5,src=0):

        # class attributes related to the model and the video
        self.src = src
        self.mp_face_detection = mp.solutions.face_detection
        self.model = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)
        self.recognition = tf.keras.models.load_model(model)
        self.capture = cv2.VideoCapture(src)
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._, self.img = self.capture.read()
        self.cTime = 0 
        self.pTime = 0
        self.path = path
        self.images = os.listdir(self.path)
        self.names = map(lambda x: x.split('.')[0], self.images)
        self.names = list(self.names)
        self.label = []
        self.mode = mode
        self.thresh = thresh
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0

        self.detected_faces = []

        for image in self.images:
            img = self.face_detection(os.path.join(self.path,image))
            self.detected_faces.append(img)

        self.embeddings = self.recognition.predict(np.array(self.detected_faces))

        # Thread for running the CNN classification of each video frame
        self.t = Thread(target=self.detectFaces)
        self.t.daemon = True
        self.t.start()

    def detection(self,img, faces):
        for face in faces:
            x1,y1 = face['keypoints']['left_eye']
            x2,y2 = face['keypoints']['right_eye']
            x,y,w,h = face['box']
        img = img[y:y+h,x:x+w,:]
        img = img/255.
        img = cv2.resize(img,(160,160))
        return img

    def face_detection(self,image):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detector = MTCNN()
        faces = detector.detect_faces(img)
        img = self.detection(img, faces)
        return img

    def euclidean(self,x1,y1,x2,y2):
        return ((x1-x2)**2 + (y1-y2)**2)**0.5

    # Thread function to detect faces in each video frame using haar cascades
    # Heavy video processing functionality should be defined here
    def detectFaces(self):

        while True:
            img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            try:
                img = img[self.y:self.y+self.h,self.x:self.x+self.w,:]
                img = img/255.
                img = cv2.resize(img,(160,160))
                v_emb = self.recognition.predict(np.expand_dims(img,axis=0))
                distances = L2(self.embeddings, v_emb)
                index = np.argmin(distances,axis=0)
                r_index = index[0] if distances[index] < self.thresh else 'Unknown' 
                try:
                    self.label.append(self.names[r_index])
                except TypeError:
                    self.label.append(r_index)
            except:
                pass
            time.sleep(1/60)
        return

    # Running the read/display of the video on the main thread
    def display(self,box_color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (255, 255, 255)
        thickness = 3

        while True:  

            self.img = cv2.flip(self.capture.read()[1],1)
            label = 'Unknown'
            try:
                label = st.mode(self.label[-self.mode:])[0][0]
                (width, height),b = cv2.getTextSize(label, font, fontScale, thickness)
            except:
                (width, height),b = cv2.getTextSize(label, font, fontScale, thickness)

            results = self.model.process(self.img)
            if results.detections:
                for detection in results.detections:
                    self.x = int(detection.location_data.relative_bounding_box.xmin*self.width)
                    self.y = int(detection.location_data.relative_bounding_box.ymin*self.height)
                    self.w = int(detection.location_data.relative_bounding_box.width*self.width)
                    self.h = int(detection.location_data.relative_bounding_box.height*self.height)
                    cv2.rectangle(self.img, (self.x, self.y), (self.x+self.w, self.y+self.h), box_color, 2)  
                    cv2.rectangle(self.img, (self.x, self.y-height), (self.x+width, self.y), box_color, -5)
                    cv2.putText(self.img, label, (self.x,self.y), font, fontScale, color, thickness, cv2.LINE_AA)

            self.cTime = time.time()
            fps = 1 / (self.cTime - self.pTime)
            self.pTime = self.cTime
            cv2.putText(self.img, "FPS: "+str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow('Video', self.img)  
            k = cv2.waitKey(20) & 0xff  
            if k==27:  
                break

        self.capture.release()

if __name__ == '__main__':
    
    video = FaceDetection('models/face_embed.h5','images',7,1.6,0)
    try:
        video.display((0,0,0))
    except Exception as e:
        print(e)
        pass