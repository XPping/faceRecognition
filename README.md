# faceRecognition
python3 + pytorch + dlib + python-opencv

# Database
VggfaceV2

# Model
We adapt the DeepFace except face align. When face align, we used the landmark detected by dlib according to the geometry of the face.

# Train
cmd: python train.py
In data set processing, the random combination of two images of the same person is labeled as 1, while the random combination of two images of different people is labeled as 0.The negative class is about twice as large as the positive class

# Test
cmd: python test.py 
I used two pictures of myself with glasses and without eyes, and the model could identify the same person. You can test it yourself.

