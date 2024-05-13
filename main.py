import math, shutil, os, time, argparse
import numpy as np
import scipy.io as sio

import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from ITrackerData import ITrackerData
from ITrackerModel import ITrackerModel

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='iTracker-pytorch-Trainer.')
parser.add_argument('--doLoad', type=str2bool, nargs='?', const=True, default=False, help="Start from scratch (do not load).")
parser.add_argument('--checkpoint', type=str, nargs='?', default=None, help="Start from this checkpoint.")
args = parser.parse_args()

# Change there flags to control what happens.
doLoad = args.doLoad # Load checkpoint at the beginning
checkpoint = args.checkpoint

workers = 8
epochs = 10
# batch_size = torch.cuda.device_count()*100 # Change if out of cuda memory
batch_size = 30

base_lr = 0.00001
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
prec1 = 0
best_prec1 = 1e20
lr = base_lr

count_test = 0
count = 0

doTest = False

CHECKPOINTS_PATH = '/workspace-cloud/koki.murata/MPIITracker/checkpoints'

def main():
    print(f'batchsize:{batch_size}')
    global args, best_prec1, weight_decay, momentum

    model = ITrackerModel()
    # model = torch.nn.DataParallel(model)
    model.cuda()
    imSize=(224,224)
    cudnn.benchmark = True   

    epoch = 0
    if doLoad:
        try:
            cp_time, fold_saved, epoch_saved = checkpoint.split('/')
            fold_saved = fold_saved.split('_')[1]   
            epoch_saved = epoch_saved.split('.')[0]
            print(fold_saved)
            print(epoch_saved)
            saved = load_checkpoint(checkpoint)
            
            
        except Exception as e:
            print(f'エラー: {e}')
            sys.exit(1)
        if saved:
            print('Loading checkpoint for epoch %05d with loss %.5f (which is the mean squared error not the actual linear error)...' % (saved['epoch'], saved['best_prec1']))
            state = saved['state_dict']
            try:
                model.module.load_state_dict(state)
            except:
                model.load_state_dict(state)
            epoch = saved['epoch']
            print(f'starting next of ep:{epoch}')
            best_prec1 = saved['best_prec1']
        else:
            print('Warning: Could not read checkpoint!')

    labelpathlist = os.listdir("/workspace-cloud/koki.murata/MPIITracker/datasets/dlib1/Label")
    labelpathlist.sort()
    impath = "/workspace-cloud/koki.murata/MPIITracker/datasets/dlib1/Image"
    train_size = 0.9  # 80%をトレーニングセットに使用
    
    #最初から
    epoch = 0
    
    
    
    
    for fold in range(1,2):
        data = ITrackerData(labelpathlist, impath, fold=fold)
        
        print(f'data{len(data)}')
        dataTrain, dataVal = train_test_split(data, train_size=train_size, random_state=42)
        print(f'train:{len(dataTrain)}, val:{len(dataVal)}')
    
        train_loader = torch.utils.data.DataLoader(
            dataTrain,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            dataVal,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)


        criterion = nn.MSELoss().cuda()

        optimizer = torch.optim.SGD(model.parameters(), lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
        
        
        pltlosslist = []
        pltiterlist = []

        # Quick test
        if doTest:
            validate(val_loader, model, criterion, epoch)
            return

        for epoch in range(0, epoch):
            adjust_learning_rate(optimizer, epoch)
            
        for epoch in range(epoch, epochs):
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, pltiterlist, pltlosslist)

            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion, epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 < best_prec1
            best_prec1 = min(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=f'checkpoint{fold}.pth.tar')


def train(train_loader, model, criterion,optimizer, epoch, pltiterlist, pltlosslist):
    global count
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    

    # switch to train mode
    model.train()

    end = time.time()

    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)
        imFace = imFace.cuda()
        imEyeL = imEyeL.cuda()
        imEyeR = imEyeR.cuda()
        faceGrid = faceGrid.cuda()
        gaze = gaze.cuda()
        
        imFace = torch.autograd.Variable(imFace, requires_grad = True)
        imEyeL = torch.autograd.Variable(imEyeL, requires_grad = True)
        imEyeR = torch.autograd.Variable(imEyeR, requires_grad = True)
        faceGrid = torch.autograd.Variable(faceGrid, requires_grad = True)
        gaze = torch.autograd.Variable(gaze, requires_grad = False)

        # compute output
        output = model(imFace, imEyeL, imEyeR, faceGrid)

        loss = criterion(output, gaze)
        
        
        pltloss = loss
        
        losses.update(loss.data.item(), imFace.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        count=count+1
        
        # 損失のプロット
        plt.clf()
        pltloss.cpu()
        pltloss = float(pltloss)
        pltlosslist.append(pltloss)
        pltiterlist.append(i + (epoch) * len(train_loader))
        plt.subplot(2, 1, 1)  # 2行1列の1番目のサブプロット
        plt.plot(pltiterlist, pltlosslist, label='Loss')

        # グラフにタイトルと軸ラベルを追加
        plt.title('loss_x')
        plt.xlabel('Total Iteration')
        plt.ylabel('Loss')

        # 凡例を表示
        plt.legend()
        
        # グラフを保存
        plt.savefig('loss_plt.png')

        print('Epoch (train): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def validate(val_loader, model, criterion, epoch):
    global count_test
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()


    oIndex = 0
    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        imFace = imFace.cuda()
        imEyeL = imEyeL.cuda()
        imEyeR = imEyeR.cuda()
        faceGrid = faceGrid.cuda()
        gaze = gaze.cuda()
        
        imFace = torch.autograd.Variable(imFace, requires_grad = False)
        imEyeL = torch.autograd.Variable(imEyeL, requires_grad = False)
        imEyeR = torch.autograd.Variable(imEyeR, requires_grad = False)
        faceGrid = torch.autograd.Variable(faceGrid, requires_grad = False)
        gaze = torch.autograd.Variable(gaze, requires_grad = False)

        # compute output
        with torch.no_grad():
            output = model(imFace, imEyeL, imEyeR, faceGrid)

        loss = criterion(output, gaze)
        
        lossLin = output - gaze
        # print(f'output:{type(output)}{output}')
        # print(f'output:{type(gaze)}{gaze}')
        lossLin = torch.mul(lossLin,lossLin)
        lossLin = torch.sum(lossLin,1)
        lossLin = torch.mean(torch.sqrt(lossLin))

        losses.update(loss.data.item(), imFace.size(0))
        lossesLin.update(lossLin.item(), imFace.size(0))
     
        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        print('Epoch (val): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                    epoch, i, len(val_loader), batch_time=batch_time,
                   loss=losses,lossLin=lossesLin))

    return lossesLin.avg



def load_checkpoint(filename):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.isdir(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH, 0o777)
    bestFilename = os.path.join(CHECKPOINTS_PATH, 'best_' + filename)
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestFilename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


if __name__ == "__main__":
    main()
    print('DONE')
