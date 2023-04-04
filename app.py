from face_recognition import FaceRecognition

video = FaceRecognition(model='models/face.h5',path='images',mode=7,thresh=1.6,src=0)
try:
    video.display(box_color=(0,0,0))
except Exception as e:
    print(e)