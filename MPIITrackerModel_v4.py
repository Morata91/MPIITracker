import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.parallel
import torch.optim
import torch.utils.data

import torchvision


'''
train_v7で使用可能


'''



block = torchvision.models.resnet.Bottleneck
layers = [3, 4, 6, 3]


class MPIITrackerModel(nn.Module):
    
    


    def __init__(self, xbins_num, ybins_num, gridSize = 25):
        super(MPIITrackerModel, self).__init__()
        
        # self.features = nn.Sequential(#ITrackerImageModelに相当
        #     nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
        #     nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
        #     nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        # )
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))#新キャラ
        
        
        
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


       # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
                
        # self.features1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     self._make_layer(torchvision.models.resnet.Bottleneck, 64, 2),
        #     self._make_layer(torchvision.models.resnet.Bottleneck, 128, 2, stride=2),
        #     self._make_layer(torchvision.models.resnet.Bottleneck, 256, 2, stride=2),
        #     self._make_layer(torchvision.models.resnet.Bottleneck, 512, 2, stride=2),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        # )
        
        self.eyesFC = nn.Linear(2 * 512 * torchvision.models.resnet.Bottleneck.expansion, 128)
        self.faceFC = nn.Linear(512 * torchvision.models.resnet.Bottleneck.expansion, 128)
        
        
        # self.eyesFC = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(in_features=2*9216, out_features=4096, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(in_features=4096, out_features=4096, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=4096, out_features=128, bias=True)
        # )
        
        # self.faceFC = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(in_features=9216, out_features=4096, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(in_features=4096, out_features=4096, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=4096, out_features=128, bias=True)
        # )
    
        
        self.gridFC = nn.Sequential(
            nn.Linear(gridSize * gridSize, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            )
       
        
        
        self.fc_x = nn.Sequential(
            nn.Linear(128+128+128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, xbins_num),
            )
        
        
        
        self.fc_y = nn.Sequential(
            nn.Linear(128+128+128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, ybins_num),
            )
        
        # 重みの初期化
        self.init_weights()
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def init_weights(self):
        
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)

    def forward(self, faces, eyesLeft, eyesRight, faceGrids):
        ##目画像ネットワーク
        x = self.conv1(eyesLeft)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        EyeL = x.view(x.size(0), -1)
        # EyeL = self.features1(eyesLeft)
        # print(EyeL.shape)
        # EyeL = self.features(eyesLeft)
        # print(EyeL.shape)
        # EyeL = self.avgpool(EyeL)
        # EyeL = EyeL.view(EyeL.size(0), -1)
        # EyeR = self.features1(eyesRight)
        # EyeR = self.features(eyesRight)
        # EyeR = self.avgpool(EyeR)
        x = self.conv1(eyesRight)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        EyeR = x.view(x.size(0), -1)
        # EyeR = EyeR.view(EyeR.size(0), -1)
        ##leftとrightをcat
        Eyes = torch.cat((EyeL, EyeR), 1)
        # print(Eyes.shape)

        ## 顔画像ネットワーク
        # Face = self.features1(faces)
        # Face = self.features(faces)
        # Face = self.avgpool(Face)
        # Face = Face.view(Face.size(0), -1)
        x = self.conv1(faces)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        Face = x.view(x.size(0), -1)
        
        ## マスク画像全結合層
        Grid = faceGrids.view(faceGrids.size(0), -1)
        
        
        
        
        ##x出力層
        xEyes = self.eyesFC(Eyes)
        xFace = self.faceFC(Face)
        xGrid = self.gridFC(Grid)

        ##y出力層
        yEyes = self.eyesFC(Eyes)
        yFace = self.faceFC(Face)
        yGrid = self.gridFC(Grid)

        # Cat all
        x = torch.cat((xEyes, xFace, xGrid), 1)
        y = torch.cat((yEyes, yFace, yGrid), 1)
        # x = self.fc(x)
        # print(f'x:{x.shape}')
        
        pre_x = self.fc_x(x)
        pre_y = self.fc_y(y)
        
        pre_x1 = 0
        
        return pre_x, pre_y, pre_x1
