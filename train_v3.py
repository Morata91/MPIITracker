import math, shutil, os, time, argparse
import numpy as np
import scipy.io as sio

import sys
from datetime import datetime

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

from ITrackerModel_v2 import ITrackerModel

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
parser.add_argument('--fold', type=int, nargs='?', default=None, help="train for this fold.")
args = parser.parse_args()

# Change there flags to control what happens.
doLoad = args.doLoad # Load checkpoint at the beginning
checkpoint_path = args.checkpoint

workers = 8
epochs = 30
batch_size = torch.cuda.device_count()*100 # Change if out of cuda memory

base_lr = 0.0001
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
prec1 = 0
best_prec1 = 1e20
lr = base_lr

count_test = 0
count = 0

gpu = 0

CHECKPOINTS_PATH = 'checkpoints'
LOG_PATH = 'trainlog'

def main():
    print(f'batchsize:{batch_size}')
    global args, best_prec1, weight_decay, momentum

    model = ITrackerModel()
    # model = torch.nn.DataParallel(model)
    model.cuda(gpu)
    imSize=(224,224)
    cudnn.benchmark = True   

    starting_epoch = 0
    starting_fold = 0
    
    now = datetime.now()
    # cp_time = f'{now.year}{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}'
    cp_time = '1214'
    print(f'project_name:{cp_time}')
    
    
    if doLoad:
        try:
            # checkpoint_path=YYMMDDHHMM/fold_XX/YY.pth.tar
            cp_time, fold_saved, epoch_saved = checkpoint_path.split('/')
            fold_saved = fold_saved.split('_')[1]   
            epoch_saved = epoch_saved.split('.')[0]
            starting_fold = int(fold_saved)
            saved = load_checkpoint(checkpoint_path)
            
            
        except Exception as e:
            print(f'エラー: {e}, checkpointが読み込めませんでした.')
            sys.exit(1)
        if saved:
            print('Loading checkpoint for epoch %05d with loss %.5f (which is the mean squared error not the actual linear error)...' % (saved['epoch'], saved['best_prec1']))
            state = saved['state_dict']
            try:
                model.module.load_state_dict(state)
            except:
                model.load_state_dict(state)
            starting_epoch = saved['epoch']
            print(f'starting next of [fold{fold_saved}/epoch:{starting_epoch}]')
            best_prec1 = saved['best_prec1']
        else:
            print('Warning: Could not read checkpoint!')

    labelpathlist = os.listdir("datasets/Label")
    labelpathlist.sort()
    impath = "datasets/Image"
    train_size = 0.9  # 80%をトレーニングセットに使用
    
    
    
    starting_fold = args.fold
    cp_time = '2222'
    
    for fold in range(starting_fold,starting_fold+1):
        log_path = os.path.join(LOG_PATH, cp_time, f'fold_{fold:02d}')
        if not os.path.isdir(log_path):
            os.makedirs(log_path, 0o777)
        log_file_path=os.path.join(log_path, f'fold_{fold:02d}.log')
            
        data = ITrackerData(labelpathlist, impath, fold=fold)
        
        print(f'all data:{len(data)}')
        #時間かかるから変えたい
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




        optimizer = torch.optim.SGD(model.parameters(), lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)

        # Quick test
        # if doTest:
        validate(val_loader, model, starting_epoch, fold, log_file_path)
        # return
        
        
        for epoch in range(0, starting_epoch):
            adjust_learning_rate(optimizer, epoch)
            
        for epoch in range(starting_epoch, epochs):
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train(train_loader, model,  optimizer, epoch, epochs, fold, log_file_path)

            # evaluate on validation set
            prec1 = validate(val_loader, model,  epoch, fold, log_file_path)

            # remember best prec@1 and save checkpoint
            is_best = prec1 < best_prec1
            best_prec1 = min(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, epoch=epoch, cp_time=cp_time, fold=fold)
            
        starting_epoch = 0
            


def train(train_loader, model, optimizer, epoch, epochs, fold, log_file_path):
    global count
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_x = AverageMeter()
    losses_y = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze) in enumerate(train_loader):
        with open(log_file_path, "a") as logfile:
    
        
            # measure data loading time
            data_time.update(time.time() - end)
            imFace = imFace.cuda(gpu)
            imEyeL = imEyeL.cuda(gpu)
            imEyeR = imEyeR.cuda(gpu)
            faceGrid = faceGrid.cuda(gpu)
            gaze = gaze.cuda(gpu)
            gaze_x = gaze[:,0].cuda(gpu)
            gaze_y = gaze[:,1].cuda(gpu)
            
            imFace = torch.autograd.Variable(imFace, requires_grad = True)
            imEyeL = torch.autograd.Variable(imEyeL, requires_grad = True)
            imEyeR = torch.autograd.Variable(imEyeR, requires_grad = True)
            faceGrid = torch.autograd.Variable(faceGrid, requires_grad = True)
            gaze = torch.autograd.Variable(gaze, requires_grad = False)
            gaze_x = torch.autograd.Variable(gaze_x, requires_grad = False)
            gaze_y = torch.autograd.Variable(gaze_y, requires_grad = False)

            # compute output
            pre_x, pre_y = model(imFace, imEyeL, imEyeR, faceGrid)
            
            loss_reg_x = reg_criterion(pre_x, gaze_x)
            loss_reg_y = reg_criterion(pre_y, gaze_y)

            
            losses_x.update(loss_reg_x.data.item(), imFace.size(0))
            losses_y.update(loss_reg_y.data.item(), imFace.size(0))
            
            loss_seq = [loss_reg_x, loss_reg_y]
            grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]

            # compute gradient and do SGD step
            optimizer.zero_grad()
            # loss_seq.backward()
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            count=count+1
            
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
            
            logger = f'{formatted_now} (train): fold[{fold}] epoch[{epoch+1}/{epochs}] iter[{i+1}/{len(train_loader)}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tData {data_time.val:.3f} ({data_time.avg:.3f})\tLoss x {losses_x.val:.4f} ({losses_x.avg:.4f}) y {losses_y.val:.4f} ({losses_y.avg:.4f})\t'
            logfile.write(logger+'\n')
            print(logger)

def validate(val_loader, model,  epoch, fold, log_file_path):
    criterion = nn.MSELoss().cuda(gpu)
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
        with open(log_file_path, "a") as logfile:
            # measure data loading time
            data_time.update(time.time() - end)
            imFace = imFace.cuda(gpu)
            imEyeL = imEyeL.cuda(gpu)
            imEyeR = imEyeR.cuda(gpu)
            faceGrid = faceGrid.cuda(gpu)
            gaze = gaze.cuda(gpu)
            
            imFace = torch.autograd.Variable(imFace, requires_grad = False)
            imEyeL = torch.autograd.Variable(imEyeL, requires_grad = False)
            imEyeR = torch.autograd.Variable(imEyeR, requires_grad = False)
            faceGrid = torch.autograd.Variable(faceGrid, requires_grad = False)
            gaze = torch.autograd.Variable(gaze, requires_grad = False)

            # compute output
            with torch.no_grad():
                pre_x, pre_y = model(imFace, imEyeL, imEyeR, faceGrid)
                
            
            output = torch.cat((pre_x, pre_y), 1)

            loss = criterion(output, gaze)
            
            lossLin = output - gaze
            
            #正解と推定データ表示
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
            
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

            logger = f'{formatted_now} (val): fold[{fold}] epoch[{epoch+1}] iter[{i+1}/{len(val_loader)}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tLoss {losses.val:.4f} ({losses.avg:.4f})\tError L2 {lossesLin.val:.4f} ({lossesLin.avg:.4f})'
            logfile.write(logger+'\n')
            print(logger)
        
            
    return lossesLin.avg

 

def load_checkpoint(filename):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(f'checkpoint file:{filename}')
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state

def save_checkpoint(state, is_best, epoch, cp_time, fold):
    save_path = os.path.join(CHECKPOINTS_PATH, cp_time, f'fold_{fold:02d}')
    filename=f'{epoch+1:02d}.pth.tar'
    if not os.path.isdir(save_path):
        os.makedirs(save_path, 0o777)
    bestFilename = os.path.join(save_path, 'best.pth.tar')
    filename = os.path.join(save_path, filename)
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
