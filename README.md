# Real Time Face Recognition

<div class='alert alert-info'>A <strong>facial recognition system</strong> is a technology capable of matching a human face from a digital image or a video frame against a database of faces. Such a system is typically employed to authenticate users through ID verification services, and works by pinpointing and measuring facial features from a given image.</div>

![This is an image](https://www.thalesgroup.com/sites/default/files/database/assets/images/2020-07/gov-facial-recognition-in-action.jpg)

## Typical Tasks in Face Recognition
- Generally Speaking, a facial recognition system can be used to accomplish two kinds of tasks
    - **Face Verification:** One-to-one match that compares a query face image against a template face image whose identity is being claimed.
    - **Face Identification:** One-to-many matches that compare a query face image against all the template images in the database to determine the identity of the query face.

![This is an image](https://drek4537l1klr.cloudfront.net/elgendy/v-8/Figures/10_img_0003.png)

## Model Training

Model Code: https://www.kaggle.com/code/quadeer15sh/face-recognition-siamese-network-triplet-loss
Due to github storage restrictions the model cannot be uploaded to the repository. Once the model is trained place them in the models directory.

## Inference Pipeline of Face Recognition Systems

1. Read an image
    - Detect face, perform alignment
2. Extract face embeddings
3. Calculate distance between embeddings present in the database and extracted input face embeddings
4. Find the index of the lowest distance, and check if distance is less than threshold
    - If yes then use the index to find the person from the database
    - Else the person in the input image is not present in the database
5. Use the OpenCV utilities to tag the person in the image

## Result

https://user-images.githubusercontent.com/38568261/229374265-7722f99d-d34b-4668-8bd8-c4306c63cc2e.mp4

## Python libraries required

```
pip install -r requirements.txt
```
