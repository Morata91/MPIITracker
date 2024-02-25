import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.parallel
import torch.optim
import torch.utils.data


'''
train_v5で使用可能


'''


class MPIITrackerModel(nn.Module):


    def __init__(self, xbins_num, ybins_num, gridSize = 25):
        super(MPIITrackerModel, self).__init__()
        
        self.features = nn.Sequential(#ITrackerImageModelに相当
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))#新キャラ
        
        
        self.eyesFC = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=2*9216, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=128, bias=True)
        )
        
        self.faceFC = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=9216, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=128, bias=True)
        )
    
        
        self.gridFC = nn.Sequential(
            nn.Linear(gridSize * gridSize, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            )
        
        self.eyegridFC = nn.Sequential(
            nn.Linear(64*64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            )
        
        
        self.fc_x = nn.Sequential(
            nn.Linear(128+128+128+128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, xbins_num),
            )
        
        
        
        self.fc_y = nn.Sequential(
            nn.Linear(128+128+128+128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, ybins_num),
            )
        
        # 重みの初期化
        self.init_weights()
        
    def init_weights(self):
        
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)

    def forward(self, faces, eyesLeft, eyesRight, faceGrids, eyeGrids):
        ##目画像ネットワーク
        EyeL = self.features(eyesLeft)
        EyeL = self.avgpool(EyeL)
        EyeL = EyeL.view(EyeL.size(0), -1)
        EyeR = self.features(eyesRight)
        EyeR = self.avgpool(EyeR)
        EyeR = EyeR.view(EyeR.size(0), -1)
        ##leftとrightをcat
        Eyes = torch.cat((EyeL, EyeR), 1)
        # print(Eyes.shape)
        # print(Eyes.shape)

        ## 顔画像ネットワーク
        Face = self.features(faces)
        Face = self.avgpool(Face)
        Face = Face.view(Face.size(0), -1)
        
        ## マスク画像全結合層
        Grid = faceGrids.view(faceGrids.size(0), -1)
        EyeGrid = eyeGrids.view(eyeGrids.size(0), -1)
        
        
        # print(Eyes.shape)
        ##x出力層
        xEyes = self.eyesFC(Eyes)
        xFace = self.faceFC(Face)
        xGrid = self.gridFC(Grid)
        xEyeGrid = self.eyegridFC(EyeGrid)

        # print('a')
        ##y出力層
        yEyes = self.eyesFC(Eyes)
        yFace = self.faceFC(Face)
        yGrid = self.gridFC(Grid)
        yEyeGrid = self.eyegridFC(EyeGrid)

        # Cat all
        x = torch.cat((xEyes, xFace, xGrid, xEyeGrid), 1)
        y = torch.cat((yEyes, yFace, yGrid, yEyeGrid), 1)
        # x = self.fc(x)
        # print(f'x:{x.shape}')
        
        pre_x = self.fc_x(x)
        pre_y = self.fc_y(y)
        
        pre_x1 = 0
        
        return pre_x, pre_y, pre_x1
