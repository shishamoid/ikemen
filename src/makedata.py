#import opencv
import dlib
import cv2
import numpy

# -*- coding: utf-8 -*-

print("pokemon")

def makedata():

    image_path = "../data/test.png"

    detector = dlib.get_frontal_face_detector()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    print(image)
    rects = detector(image, 2)
    # rectsの数だけ顔を検出
    PREDICTOR_PATH = '../model/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    for rect in rects:
        landmarks = numpy.matrix(
        [[p.x, p.y] for p in predictor(image, rect).parts()]
        )
        print(landmarks[0:17])

makedata()
#print(landmarks)
#print("pokemon")
