import argparse
import os
import shutil
import time, math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import torch.utils.model_zoo as model_zoo
from torch.autograd.variable import Variable


## xとyに分ける


'''
Pytorch model for the iTracker.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''


momentum = 0.9
eps = 1e-5


class ItrackerImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self):
        super(ItrackerImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96, momentum=momentum, eps=eps),  # バッチ正規化を追加
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256, momentum=momentum, eps=eps),  # バッチ正規化を追加
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, momentum=momentum, eps=eps),  # バッチ正規化を追加
            
        )
        
        # 重みの初期化
        self.init_weights()
        
    def init_weights(self):
        # nn.Sequential内の各モジュールの重みを初期化
        for layer in self.features:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                # init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class FaceImageModel(nn.Module):
    
    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.conv = ItrackerImageModel()
        self.fc = nn.Sequential(
            nn.Linear(12*12*64, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128, momentum=momentum, eps=eps),  # バッチ正規化を追加
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64, momentum=momentum, eps=eps),  # バッチ正規化を追加
            )
        
        # 重みの初期化
        self.init_weights()
        
    def init_weights(self):
        # nn.Sequential内の各モジュールの重みを初期化
        for layer in self.fc:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                # init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class FaceGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize = 25):
        super(FaceGridModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),#128-->64
            nn.ReLU(inplace=True),
            )
        
        # 重みの初期化
        self.init_weights()
        
    def init_weights(self):
        # nn.Sequential内の各モジュールの重みを初期化
        for layer in self.fc:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                # init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class ITrackerModel(nn.Module):


    def __init__(self):
        super(ITrackerModel, self).__init__()
        self.eyeModel = ItrackerImageModel()
        self.faceModel = FaceImageModel()
        self.gridModel = FaceGridModel()
        
        # Fully connected layer for eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(2 * 12 * 12 * 64, 64),#一番右128-->64
            nn.ReLU(inplace=True),
        )

        # Separate fully connected layers for x and y coordinates
        self.fc_x = nn.Sequential(
            nn.Linear(64 + 64 + 64, 128),#一番左と一番右128-->64
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),  # Output for x coordinate
        )

        self.fc_y = nn.Sequential(
            nn.Linear(64 + 64 + 64, 128),#一番左と一番右128-->64
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),  # Output for y coordinate
        )

        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        for layer in self.fc_x:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                # init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        for layer in self.fc_y:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                # init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, faces, eyesLeft, eyesRight, faceGrids):
        # Eye nets
        xEyeL = self.eyeModel(eyesLeft)
        xEyeR = self.eyeModel(eyesRight)
        # Cat and FC
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesFC(xEyes)
        print(xEyes.shape)

        # Face net
        xFace = self.faceModel(faces)
        xGrid = self.gridModel(faceGrids)

        # Cat all
        x = torch.cat((xEyes, xFace, xGrid), 1)

        # Separate predictions for x and y coordinates
        x_pred = self.fc_x(x)
        y_pred = self.fc_y(x)
        

        return x_pred, y_pred
