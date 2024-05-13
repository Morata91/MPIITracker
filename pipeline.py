import pathlib
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

import torchvision.transforms as transforms

from results import GazeResultContainer

import dlib
import os

from MPIITrackerModel_v6 import MPIITrackerModel

from PIL import Image

x_max=150
y_max=200
binwidth_x = 30
binwidth_y = 20

ORIG_IMSIZE = (1280,720)

class Pipeline:

    def __init__(
        self, 
        weights: pathlib.Path,  
        include_detector:bool = True,
        confidence_threshold:float = 0.1
        ):
        
        self.xbins_num = x_max*2//binwidth_x
        self.ybins_num = y_max//binwidth_y
        self.idx_tensor_x = [idx for idx in range(self.xbins_num)]
        self.idx_tensor_x = torch.autograd.Variable(torch.FloatTensor(self.idx_tensor_x))
        self.idx_tensor_y = [idx for idx in range(self.ybins_num)]
        self.idx_tensor_y = torch.autograd.Variable(torch.FloatTensor(self.idx_tensor_y))
        
        self.transformFace = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        self.transformEyeL = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        self.transformEyeR = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        # Save input parameters
        self.weights = weights
        self.include_detector = include_detector
        self.confidence_threshold = confidence_threshold

        # Create model
        self.model = MPIITrackerModel(self.xbins_num, self.ybins_num)
        saved = torch.load(self.weights)
        saved_state_dict = saved['state_dict']
        self.model.load_state_dict(saved_state_dict)
        self.model.eval()

        # Create facedetector
        if self.include_detector:

            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("./dlib_model/shape_predictor_68_face_landmarks.dat")

            self.softmax = nn.Softmax(dim=1)

    def step(self, image: np.ndarray, width, height) -> GazeResultContainer:

        # Creating containers
        face_imgs = []
        xs = []
        ys = []
        ws = []
        hs = []
        lxs = []
        lys = []
        lws = []
        lhs = []
        rxs = []
        rys = []
        rws = []
        rhs = []
        leye_imgs = []
        reye_imgs = []

        landmarks = []
        scores = []
        
        self.width = width
        self.height = height

        if self.include_detector:

            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)
                # confidence = self.detector(image, 0)[0].confidence
                for face in faces:
                    # print(face)

                    # Apply threshold
                    # if confidence < self.confidence_threshold:
                    #     print(confidence)
                    #     continue
                    # print('a')

                    # 顔領域を切り取る
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    
                    # Crop image
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face_roi = image[y:y + h, x:x + w]
                    resized_face = cv2.resize(face_roi, (224, 224), interpolation=cv2.INTER_AREA)
                    resized_face = Image.fromarray(resized_face)
                    face_imgs.append(resized_face)
                    
                    shape = self.predictor(gray, face)
                    # print('a')
                    
                    lcenter_x = (shape.part(36).x + shape.part(39).x) // 2
                    lcenter_y = (shape.part(36).y + shape.part(39).y) // 2
                    left_eye_roi = image[lcenter_y-40:lcenter_y+40, lcenter_x-40:lcenter_x+40]
                    resized_left_eye = cv2.resize(left_eye_roi, (224, 224), interpolation=cv2.INTER_AREA)
                    cv2.imwrite('a.png', resized_left_eye)
                    resized_left_eye = Image.fromarray(resized_left_eye)
                    leye_imgs.append(resized_left_eye)
                    lx = shape.part(36).x
                    ly = (shape.part(37).y + shape.part(38).y) // 2
                    lw = shape.part(39).x - lx
                    lh = (shape.part(40).y + shape.part(41).y) // 2 - ly
                    
                    rcenter_x = (shape.part(42).x + shape.part(45).x) // 2
                    rcenter_y = (shape.part(42).y + shape.part(45).y) // 2
                    right_eye_roi = image[rcenter_y-40:rcenter_y+40, rcenter_x-40:rcenter_x+40]
                    resized_right_eye = cv2.resize(right_eye_roi, (224, 224), interpolation=cv2.INTER_AREA)
                    resized_right_eye = Image.fromarray(resized_right_eye)
                    reye_imgs.append(resized_right_eye)
                    rx = shape.part(42).x
                    ry = (shape.part(43).y + shape.part(44).y) // 2
                    rw = shape.part(45).x - lx
                    rh = (shape.part(46).y + shape.part(47).y) // 2 - ly
                    
                    
                    # # Save data
                    # xs.append(x)
                    # ys.append(y)
                    # ws.append(w)
                    # hs.append(h)
                    # lxs.append(lx)
                    # lys.append(ly)
                    # lws.append(lw)
                    # lhs.append(lh)
                    # rxs.append(rx)
                    # rys.append(ry)
                    # rws.append(rw)
                    # rhs.append(rh)
                    # print(type(resized_face))
                    
                    # resized_face = torch.tensor(resized_face)
                    # resized_left_eye = torch.tensor(resized_left_eye)
                    # resized_right_eye = torch.tensor(resized_right_eye)
                    
                    imFace = self.transformFace(resized_face)
                    imEyeL = self.transformEyeL(resized_left_eye)
                    imEyeR = self.transformEyeR(resized_right_eye)
                    imFace = imFace.unsqueeze(0)
                    imEyeL = imEyeL.unsqueeze(0)
                    imEyeR = imEyeR.unsqueeze(0)
                    params = np.array([x,y,w,h])
                    faceGrid = self.makeGrid(params)
                    

                    
                    eyeParams = np.array([lx,ly,lw,lh,rx,ry,rw,rh])
                    eyeGrid = self.makeEyeGrid(params, eyeParams)
                    
                    faceGrid = torch.FloatTensor(faceGrid)
                    faceGrid = faceGrid.unsqueeze(0)
                    eyeGrid = torch.FloatTensor(eyeGrid)
                    eyeGrid = eyeGrid.unsqueeze(0)

                    # Predict gaze
                    # print(type(resized_face))
                    # print(resized_face)
                    # print(eyeGrid.shape)
                    with torch.no_grad():
                        # x, y, pre_x1 = self.model(resized_face, resized_left_eye, resized_right_eye, faceGrid, eyeGrid)
                        xp, yp, pre_x1 = self.model(imFace, imEyeL, imEyeR, faceGrid, eyeGrid)
                    
                    x_norm = self.softmax(xp)
                    y_norm = self.softmax(yp)
                    
                    x_exp = torch.sum(x_norm * self.idx_tensor_x, 1) * binwidth_x - x_max
                    y_exp = torch.sum(y_norm * self.idx_tensor_y, 1) * binwidth_y
                    
                    
                    x = x_exp.unsqueeze(1)
                    y = y_exp.unsqueeze(1)


            except:
                

                x=0
                y=0

        # else:
        #     pitch, yaw = self.predict_gaze(frame)

        # Save data
        results = GazeResultContainer(
            x=x,
            y=y
        )

        return results


    def makeGrid(self, params):
        gridLen = 25*25
        grid = np.zeros([gridLen,], np.float32)
        
        x = int(params[0] * 25 / self.width)
        y = int(params[1] * 25 / self.height)
        w = int(params[2] * 25 / self.width)
        h = int(params[3] * 25 / self.height)
        
        indsX = np.array([i % 25 for i in range(gridLen)])
        indsY = np.array([i // 25 for i in range(gridLen)])
        condX = np.logical_and(indsX >= x, indsX < x+w) 
        condY = np.logical_and(indsY >= y, indsY < y+h) 
        cond = np.logical_and(condX, condY)

        grid[cond] = 1
        return grid
    
    def makeEyeGrid(self, params, eyeParams):
        gridLen = 64*64
        grid = np.zeros([gridLen,], np.float32)
        
        lx = int((eyeParams[0] - params[0]) * 64 / params[2])
        ly = int((eyeParams[1] - params[1]) * 64 / params[3])
        lw = int(eyeParams[2] * 64 / params[2])
        lh = int(eyeParams[3] * 64 / params[3])
        
        indsX = np.array([i % 64 for i in range(gridLen)])
        indsY = np.array([i // 64 for i in range(gridLen)])
        lcondX = np.logical_and(indsX >= lx, indsX < lx+lw) 
        lcondY = np.logical_and(indsY >= ly, indsY < ly+lh) 
        lcond = np.logical_and(lcondX, lcondY)
        
        rx = int((eyeParams[4] - params[0]) * 64 / params[2])
        ry = int((eyeParams[5] - params[1]) * 64 / params[3])
        rw = int(eyeParams[6] * 64 / params[2])
        rh = int(eyeParams[7] * 64 / params[3])
        
        indsX = np.array([i % 64 for i in range(gridLen)])
        indsY = np.array([i // 64 for i in range(gridLen)])
        rcondX = np.logical_and(indsX >= rx, indsX < rx+rw) 
        rcondY = np.logical_and(indsY >= ry, indsY < ry+rh) 
        rcond = np.logical_and(rcondX, rcondY)
        
        cond = np.logical_or(lcond, rcond)

        grid[cond] = 1
        return grid