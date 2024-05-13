import torch.utils.data as data
# import scipy.io as sio
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import re

IMPATH = "datasets//dlib/Image/"
# IMPATH = "datasets/Image/"
LABELPATH = "datasets/dlib/Label"
# LABELPATH = "datasets/Label"




class SubtractMean(object):
    """Normalize an tensor image with mean.
    """

    def __init__(self, meanImg):
        self.meanImg = transforms.ToTensor()(meanImg / 255)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """       
        return tensor.sub(self.meanImg)

#データセットの作成
class ITrackerData(data.Dataset):
    def __init__(self, labelpathlist, imPath, train = True, imSize=(224,224), gridSize=(25, 25), fold=0):

        path  = labelpathlist
        self.imSize = imSize
        self.gridSize = gridSize
        self.imPath = imPath
        self.orig_list_len = 0
        self.lines = []

        print('Loading iTracker dataset...')
        
        # self.transformFace = transforms.Compose([
        #     transforms.Resize(self.imSize),
        #     transforms.ToTensor(),
        #     SubtractMean(meanImg=self.faceMean),
        # ])
        # self.transformEyeL = transforms.Compose([
        #     transforms.Resize(self.imSize),
        #     transforms.ToTensor(),
        #     SubtractMean(meanImg=self.eyeLeftMean),
        # ])
        # self.transformEyeR = transforms.Compose([
        #     transforms.Resize(self.imSize),
        #     transforms.ToTensor(),
        #     SubtractMean(meanImg=self.eyeRightMean),
        # ])
        
        self.transformFace = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor()
        ])
        self.transformEyeL = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor()
        ])
        self.transformEyeR = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor()
        ])

        ##trainの時はp00以外、testの時はp00を使用
        if train==True:
        
            path.pop(fold)
        
        else:
            path=path[fold]
            
        if isinstance(path, list):
            print(f'path{path}')
            for i, p in enumerate(path):
                with open(os.path.join(LABELPATH, p)) as f:
                    lines = f.readlines()
                    if i>=fold:
                        print(f'p{i+1}:{len(lines)}')
                    else:
                        print(f'p{i}:{len(lines)}')
                    # self.orig_list_len += len(lines)
                    
                    #開発用データセットサイズ
                    dev_size_count = 0
                    
                    for line in lines:
                        #開発用
                        dev_size_count += 1
                        if dev_size_count >= 1500:
                            break
                            
                        self.lines.append(line)
        else:
            print(os.path.join(LABELPATH,path))
            with open(os.path.join(LABELPATH,path)) as f:
                lines = f.readlines()
                
                #開発用データセットサイズ
                dev_size_count = 0
                
                for line in lines:
                    #開発用
                    dev_size_count += 1
                    if dev_size_count >= 500:
                        break
                    
                    
                #     gazex = line.strip().split(" ")[1]
                #     gazey = line.strip().split(" ")[2]
                #     label = np.array(gazex, gazey).astype("float")
                    ##以下の文があると読み込みデータが全部０になる。
                    # if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                    self.lines.append(line)
                
            

    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
            #im = Image.new("RGB", self.imSize, "white")

        return im


    def makeGrid(self, params):
        gridLen = self.gridSize[0] * self.gridSize[1]
        grid = np.zeros([gridLen,], np.float32)
        
        indsY = np.array([i // self.gridSize[0] for i in range(gridLen)])
        indsX = np.array([i % self.gridSize[0] for i in range(gridLen)])
        condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2]) 
        condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3]) 
        cond = np.logical_and(condX, condY)

        grid[cond] = 1
        return grid

    def __getitem__(self, idx):
        
        line = self.lines[idx]
        line = line.strip().split(" ")
        imFacePath = IMPATH + line[0].split("/")[0] + "/face/" + line[0].split("/")[1] + "_face.jpg"
        imEyeRPath = IMPATH + line[0].split("/")[0] + "/right/" + line[0].split("/")[1] + "_right.jpg"
        imEyeLPath = IMPATH + line[0].split("/")[0] + "/left/" + line[0].split("/")[1] + "_left.jpg"

        imFace = self.loadImage(imFacePath)
        imEyeL = self.loadImage(imEyeLPath)
        imEyeR = self.loadImage(imEyeRPath)

        imFace = self.transformFace(imFace)
        imEyeL = self.transformEyeL(imEyeL)
        imEyeR = self.transformEyeR(imEyeR)

        gaze = np.array([line[1], line[2]], np.float32)


        params = np.array([int(line[3]), int(line[4]), int(line[5]), int(line[6])])
        faceGrid = self.makeGrid(params)
        

        # to tensor
        row = torch.LongTensor([int(idx)])
        faceGrid = torch.FloatTensor(faceGrid)
        gaze = torch.FloatTensor(gaze)

        return row, imFace, imEyeL, imEyeR, faceGrid, gaze
    
        
    def __len__(self):
        return len(self.lines)
