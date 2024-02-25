import torch.utils.data as data
# import scipy.io as sio
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np


IMPATH = "datasets//dlib_addeye/Image/"
# IMPATH = "datasets/Image/"
LABELPATH = "datasets/dlib_addeye/Label"
# LABELPATH = "datasets/Label"

DEV_NUM = 100

ORIG_IMSIZE = (1280,720)




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
class MPIITrackerData(data.Dataset):
    def __init__(self, labelpathlist, imPath, train = True, imSize=(224,224), gridSize=(25, 25), fold=0, x_max=140, y_max=180, binwidth_x=5, binwidth_y=5, dev=False):

        path  = labelpathlist
        self.imSize = imSize
        self.gridSize = gridSize
        self.imPath = imPath
        self.orig_list_len = 0
        self.lines = []
        self.x_max = x_max
        self.y_max = y_max
        self.binwidth_x=binwidth_x
        self.binwidth_y=binwidth_y
        
        #binの作成
        self.bins_x = np.array(range(-1*self.x_max, self.x_max, self.binwidth_x))
        self.bins_y = np.array(range(0, self.y_max, self.binwidth_y))
        

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
            
        if isinstance(path, list):#Trainの時
            print(f'path{path}')
            for i, p in enumerate(path):
                counter = 0
                with open(os.path.join(LABELPATH, p)) as f:
                    lines = f.readlines()
                    
                    
                    
                    #開発用データセットサイズ
                    dev_size_count = 0
                    
                    for line in lines:
                        #開発用
                        if dev and dev_size_count >= DEV_NUM:
                            break
                        elif counter >= 1000:
                            break
                        dev_size_count += 1
                        
                        #binの範囲外のデータの除外
                        gazex = line.strip().split(" ")[1]
                        gazex = float(gazex)
                        gazey = line.strip().split(" ")[2]
                        gazey = float(gazey)
                        if abs(gazex) <= x_max and 0 <= gazey and gazey <= y_max:
                            self.lines.append(line)
                            counter += 1
                        # else:
                        #     print(f'{gazex}, {gazey}')
                        
                    #データ数の表示
                    if i>=fold:
                        print(f'p{i+1}\torig:{len(lines)}, use:{counter}')
                    else:
                        print(f'p{i}\torig:{len(lines)}, use:{counter}')
                        
                    
        else:#Testの時
            print(os.path.join(LABELPATH,path))
            counter = 0
            with open(os.path.join(LABELPATH,path)) as f:
                lines = f.readlines()
                
                #開発用データセットサイズ
                dev_size_count = 0
                
                for line in lines:
                    #開発用
                    if dev and dev_size_count >= DEV_NUM:
                        break
                    elif counter >= 1000:
                        break
                    dev_size_count += 1
                    
                    #binの範囲外のデータの除外
                    gazex = line.strip().split(" ")[1]
                    gazex = float(gazex)
                    gazey = line.strip().split(" ")[2]
                    gazey = float(gazey)
                    if abs(gazex) <= x_max and 0 <= gazey and gazey <= y_max:
                        self.lines.append(line)
                        counter += 1
                    # else:
                    #     print(f'{gazex}, {gazey}')
                    
                print(f'p{fold}\torig:{len(lines)}, use:{counter}')
                    
                
            

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
        
        x = int(params[0] * self.gridSize[0] / ORIG_IMSIZE[0])
        y = int(params[1] * self.gridSize[1] / ORIG_IMSIZE[1])
        w = int(params[2] * self.gridSize[0] / ORIG_IMSIZE[0])
        h = int(params[3] * self.gridSize[1] / ORIG_IMSIZE[1])
        
        indsX = np.array([i % self.gridSize[0] for i in range(gridLen)])
        indsY = np.array([i // self.gridSize[0] for i in range(gridLen)])
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
        
        eyeParams = np.array([int(line[7]), int(line[8]), int(line[9]), int(line[10]), int(line[11]), int(line[12]), int(line[13]), int(line[14])])
        eyeGrid = self.makeEyeGrid(params, eyeParams)
        

        # to tensor
        row = torch.LongTensor([int(idx)])
        faceGrid = torch.FloatTensor(faceGrid)
        gaze = torch.FloatTensor(gaze)
        
        #binのインデックスの取得
        binned_x = np.digitize(gaze[0], self.bins_x) - 1
        binned_y = np.digitize(gaze[1], self.bins_y) - 1

        return row, imFace, imEyeL, imEyeR, faceGrid, gaze, binned_x, binned_y, imFacePath, eyeGrid
    
        
    def __len__(self):
        return len(self.lines)
