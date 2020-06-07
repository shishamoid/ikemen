#import opencv
import dlib
import cv2
import numpy
import torch.nn as nn
import torch
import glob
import os,sys
import re
from torchvision import models, transforms
# -*- coding: utf-8 -*-


#https://qiita.com/pokohide/items/43203f109fd95df9a7cc
#https://qiita.com/wwacky/items/98d8be2844fa1b778323
#https://blog.chowagiken.co.jp/entry/2019/06/28/OpenCV%E3%81%A8dlib%E3%81%AE%E9%A1%94%E6%A4%9C%E5%87%BA%E6%A9%9F%E8%83%BD%E3%81%AE%E6%AF%94%E8%BC%83

#transform
#https://blog.shikoan.com/pytorch-load-different-size-image/

def makedata(datapath):
    print("start data making")

    #image_path = "../data/test.png"
    image_path = glob.glob("{}*.png".format(datapath))

    detector = dlib.get_frontal_face_detector()
    #print(len(image_path))
    images = []
    label = 1

    for i in range(len(image_path)):
         # 画像をRGB変換
        image = cv2.imread(image_path[i], cv2.IMREAD_COLOR)
        rects = detector(image, 2)
        # rectsの数だけ顔を検出
        PREDICTOR_PATH = '../model/shape_predictor_68_face_landmarks.dat'
        predictor = dlib.shape_predictor(PREDICTOR_PATH)

        for rect in rects:
            cv2.rectangle(image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 0, 0), 2)
            image = image[rect.top():rect.bottom(),rect.left():rect.right()]
            images.append(image)
            cv2.imshow("apple",image)
            landmarks = numpy.matrix(
            [[p.x, p.y] for p in predictor(image, rect).parts()]
            )

            shape = landmarks
            shape = predictor(image,rect)

            frame = image.copy()


            for shape_point_count in range(shape.num_parts):
                shape_point = shape.part(shape_point_count)
                #print(shape_point_count)
    # [0-16]:輪郭
                cv2.circle(frame, (int(shape_point.x), int(shape_point.y)), 20, (0, 0, 255), -1)
                print("ringo")
                #print(int(shape_point.x))
                #print(int(shape_point.y))
                x = int(shape_point.x)
                y = int(shape_point.y)
                #print(x,y)

         # [17-21]眉（右）
                #cv2.circle(frame,(150,204),30,(0,0,255),-1)
                print(x,y)
                cv2.circle(frame,(200,207), 20,(0, 255, 0),-1)
                # [22-26]眉（左）
                #cv2.circle(frame,(200,200),20,(0,0,255),-1)
                cv2.circle(frame,(x,y),20,(255, 0, 0),-1)
                # [27-30]鼻背
                #cv2.circle(frame,(int(shape_point.x),int(shape_point.y)),20,(0, 255, 255),-1)
                # [31-35]鼻翼、鼻尖
                #cv2.circle(frame,(int(shape_point.x),int(shape_point.y)),20,(255, 255, 0),-1)
                # [36-4142目47）
                #cv2.circle(frame,(int(shape_point.x),int(shape_point.y)),20,(255, 0, 255),-1)
                # [42-47]目（左）
                #cv2.circle(frame,(int(shape_point.x),int(shape_point.y)),20,(0, 0, 128),-1)
                # [48-54]上唇（上側輪郭）
                #cv2.circle(frame,(int(shape_point.x),int(shape_point.y)),20,(0, 128, 0),-1)
                # [54-59]下唇（下側輪郭）
                #cv2.circle(frame,(int(shape_point.x),int(shape_point.y)),20,(128, 0, 0),-1)
                # [60-64]上唇（下側輪郭）
                #cv2.circle(frame,(int(shape_point.x),int(shape_point.y)),20,(0, 128, 255),-1)
                # [65-67]下唇（上側輪郭）
                #cv2.circle(frame,(int(shape_point.x),int(shape_point.y)),20,(128, 255, 0),-1)
                #cv2.circle(frame,(int(shape.part(8).x),int(shape.part(8).y)),10,(255,255,255),-1)
                #cv2.circle(frame,(200,200),30,(0,255,255),-1)
            cv2.circle(frame,(x,y),20,(255, 0, 0),-1)
                #cv2.imshow('face landmark detector', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


            #cv2.circle(image,(447,63), 63, (0,0,255), -1)
            #cv2.imshow('img', image)

            cv2.circle(frame,(x,y),20,(255, 0, 0),-1)
            cv2.imshow('face landmark detector', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print(landmarks[0:17])
            #結果の表示
            #cv2.circle(image,(200,200),30,(0,0,255),-1)
            cv2.imshow('img', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print("finish triming")
    print(images[1].shape)

    return images, label

class mydataset(torch.utils.data.Dataset):

    def __init__(self,image_path,label):
        self.image,self.label = makedata(image_path)

    def transform(self):
        transforms.Compose([
        #ランダムでトリミング
        transforms.RandomResizedCrop(100,scale=(0.5,1.0),ratio=(1.0,1.0)),
        #ランダムで反転
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.ToTensor(),
        #正規化　平均0分散1
        torchvision.Normalize(0,1)
        ])

    def __len__(self):#必須
        return len(self.image)

    def __getitem__(self,index):#必須
        out_data = transform(self.image[index])
        out_data_label = self.label

        return out_data, out_data_label


def dataloader(datapath):
    print("data loding")
    makedata(datapath)


makedata("../data/")
#print(landmarks)
#print("pokemon")
