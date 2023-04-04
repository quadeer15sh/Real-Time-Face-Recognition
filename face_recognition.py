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

class FaceRecognition:

    """
    A class for the multiprocessing face recognition pipeline.
    ...
    Attributes
    ----------
    model : str
        String containing the path to the model
    path : str
        String containing the path to the images
    mode : int
        Mode value to calculate the rolling mode of the predicted names
    thresh : float
        threhold value for indicating the threshold within which the L2 distance is found acceptable to recognize a person's identity
    src: int/str
        video source, if 0 is given then it reads the video frames from the webcam, if string of a video path is given then the video frames are read
    recognition: object
        Tensorflow model object for a trained face embeddings model
    capture: object
        OpenCV's video capturing object
    width: int
        width of the captured video
    height: int
        height of the captured video
    cTime: int/object
        current time
    pTime: int/object
        previous time
    images: list
        list of images present in the image directory path provided
    names: list
        list of names parsed by removing the .jpg/png/jpeg extensions from the filenames in the images list
    label: list
        list containing the labels predicted for each frame of the video
    x: int
        x coordinate of the bounding box
    y: int
        y coordinate of the bounding box
    w: int
        width of the bounding box
    h: int 
        height of the bounding box
    detected_faces: list
        list containing the cropped detected faces from the image
    embeddings: numpy.ndarray
        a numpy array containing the face embeddings of all the detected faces from the list of images, these embeddings will serve as a smallscale database
    Methods
    -------
    detection(img,faces):
        Returns the cropped part of the detected face which is normalized and resized to 160x160
    face_detection(image):
        Returns the cropped image using the MTCNN face detection neural network it uses the detection method in the class to retrieve the cropped image
    recognizeFaces():
        Multithreaded function which concurrently runs and performs the face recognition using the loaded model
    display():
        Displays the recognized face label and the bounding box to localize the image
    """
    
    def __init__(self,model,path,mode,thresh=1.5,src=0):

        # class attributes related to the model and the video
        self.src = src
        self.recognition = tf.keras.models.load_model(model)
        self.capture = cv2.VideoCapture(src)
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        _, self.img = self.capture.read()
        self.cTime = 0 
        self.pTime = 0
        self.images = os.listdir(path)
        self.names = list(map(lambda x: x.split('.')[0], self.images))
        self.label = []
        self.mode = mode
        self.thresh = thresh
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0

        self.detected_faces = []

        for image in self.images:
            img = self.face_detection(os.path.join(path,image))
            self.detected_faces.append(img)

        self.embeddings = self.recognition.predict(np.array(self.detected_faces))

        # Thread for running the Face Recognition Concurrently
        self.t = Thread(target=self.recognizeFaces)
        self.t.daemon = True
        self.t.start()

    def detection(self,img, faces):
        for face in faces:
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

    # Heavy video processing functionality should be defined here
    def recognizeFaces(self):

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
            except IndexError:
                (width, height),b = cv2.getTextSize(label, font, fontScale, thickness)
            
            with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:

                results = face_detection.process(self.img)
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
    
    video = FaceRecognition(model='models/face_embed.h5',path='images',mode=7,thresh=1.6,src=0)
    try:
        video.display(box_color=(0,0,0))
    except Exception as e:
        print(e)