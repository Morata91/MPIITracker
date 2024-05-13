'''
alexnet
eyegrid追加
MPIITrackerModel_v5
MPIITrackerData_v2
でトレーニング

eval_v5でテスト?

'''



import math, shutil, os, time, argparse
# import numpy as np
# import scipy.io as sio

import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models

import matplotlib.pyplot as plt

from MPIITrackerData_v2 import MPIITrackerData
from MPIITrackerModel_v5 import MPIITrackerModel

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='iTracker-pytorch-Trainer.')
parser.add_argument('--doLoad', type=bool, nargs='?', const=True, default=False, help="Start from scratch (do not load).")
parser.add_argument('--checkpoint', type=str, nargs='?', default=None, help="Start from this checkpoint.")
parser.add_argument('--fold', type=int, nargs='?', default=None, required=True, help="train for this fold.")
parser.add_argument('--dev', type=bool, nargs='?', const=True, default=False, help="Develop Mode.")
parser.add_argument('--use', type=str, nargs='?', const=True, default=None, required=True,  help="Using device name.")
args = parser.parse_args()

# Change there flags to control what happens.
doLoad = args.doLoad # Load checkpoint at the beginning
checkpoint_path = args.checkpoint
dev = args.dev
pc_name = args.use

##ハイパーパラメータ
batch_size = 30
alpha = 1

epochs = 50
base_lr = 0.0001
lr = base_lr
momentum = 0.9
weight_decay = 1e-5
print_freq = 10
prec1 = 0
best_prec1 = 1e20
workers = 8
# batch_size = torch.cuda.device_count()*100 # Change if out of cuda memory

##使用するgpuの指定
gpu = 0

##binに関する変数の指定
x_max=150
y_max=200
binwidth_x = 30
binwidth_y = 20

CHECKPOINTS_PATH = 'checkpoints'
LOG_PATH = 'trainlog'

pltiterlist = []
pltlossxlist = []
pltlossylist = []

def main():
    print(f'batchsize:{batch_size}, lr:{lr}, alpha:{alpha}')
    xbins_num = x_max*2//binwidth_x
    ybins_num = y_max//binwidth_y
    print(f'ビン数・・・x:{xbins_num}  y:{ybins_num}')
    global args, best_prec1, weight_decay, momentum

    model = MPIITrackerModel(xbins_num, ybins_num)
    # モデルをGPUに転送
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model = torch.nn.DataParallel(model)
    model.cuda(gpu)
    cudnn.benchmark = True   

    starting_epoch = 0
    starting_fold = 0
    
    now = datetime.now()
    cp_time = f'{now.year}{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}'
    print(f'project_name:{cp_time}')
    
    
    if doLoad:
        try:
            # checkpoint_path=YYMMDDHHMM/fold_XX/YY.pth.tar
            # cp_time, fold_saved, epoch_saved = checkpoint_path.split('/')
            # fold_saved = fold_saved.split('_')[1]   
            # epoch_saved = epoch_saved.split('.')[0]
            # starting_fold = int(fold_saved)
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
            # starting_epoch = saved['epoch']
            # print(f'starting next of [fold{fold_saved}/epoch:{starting_epoch}]')
            best_prec1 = saved['best_prec1']
        else:
            print('Warning: Could not read checkpoint!')

    labelpathlist = os.listdir("datasets/Label")
    labelpathlist.sort()
    impath = "datasets/Image"
    train_size = 0.9  # 80%をトレーニングセットに使用
    
    
    #トレーニングするfoldの設定　２つ以上一気にできないので
    starting_fold = args.fold
    
    #プロジェクト名
    cp_time = '0125'
    
    for fold in range(starting_fold,starting_fold+1):
        log_path = os.path.join(LOG_PATH, cp_time, f'fold_{fold:02d}')
        if not os.path.isdir(log_path):
            os.makedirs(log_path, 0o777)
        log_file_path=os.path.join(log_path, f'fold_{fold:02d}.log')
            
        data = MPIITrackerData(labelpathlist, impath, fold=fold, x_max=x_max, y_max=y_max, binwidth_x=binwidth_x, binwidth_y=binwidth_y, dev=dev)
        
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


        # criterion_MSE = nn.MSELoss().cuda()
        # criterion_CLE = nn.CrossEntropyLoss().cuda()
        # softmax = nn.Softmax(dim=1).cuda()
        # idx_tensor_x = [idx for idx in range(xbins_num)]
        # idx_tensor_x = torch.autograd.Variable(torch.FloatTensor(idx_tensor_x)).cuda()
        # idx_tensor_y = [idx for idx in range(ybins_num)]
        # idx_tensor_y = torch.autograd.Variable(torch.FloatTensor(idx_tensor_y)).cuda()
        
        criterion_MSE = nn.MSELoss().cuda(gpu)
        criterion_CLE = nn.CrossEntropyLoss().cuda(gpu)
        softmax = nn.Softmax(dim=1).cuda(gpu)
        idx_tensor_x = [idx for idx in range(xbins_num)]
        idx_tensor_x = torch.autograd.Variable(torch.FloatTensor(idx_tensor_x)).cuda(gpu)
        idx_tensor_y = [idx for idx in range(ybins_num)]
        idx_tensor_y = torch.autograd.Variable(torch.FloatTensor(idx_tensor_y)).cuda(gpu)

        # optimizer = torch.optim.SGD(model.parameters(), lr,
        #                             momentum=momentum,
        #                             weight_decay=weight_decay)
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr)

        # Quick test
        # if doTest:
        validate(val_loader, model, criterion_CLE, criterion_MSE, softmax, idx_tensor_x, idx_tensor_y, starting_epoch, fold, log_file_path)
        # return
        
        
        for epoch in range(0, starting_epoch):
            adjust_learning_rate(optimizer, epoch)
            
        for epoch in range(starting_epoch, epochs):
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train(train_loader, model, criterion_CLE, criterion_MSE, softmax, idx_tensor_x, idx_tensor_y, optimizer, epoch, epochs, fold, log_path)

            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion_CLE, criterion_MSE, softmax, idx_tensor_x, idx_tensor_y, epoch, fold, log_file_path)

            # remember best prec@1 and save checkpoint
            is_best = prec1 < best_prec1
            best_prec1 = min(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, epoch=epoch, cp_time=cp_time, fold=fold)
            
        starting_epoch = 0
            


def train(train_loader, model, criterion_CLE, criterion_MSE, softmax, idx_tensor_x, idx_tensor_y, optimizer, epoch, epochs, fold, log_path):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Total_losses_x = AverageMeter()
    Total_losses_y = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    
    #損失プロット用
    iterations_per_epoch = 100
    total_iterations = epochs * iterations_per_epoch
    
    log_file_path=os.path.join(log_path, f'fold_{fold:02d}.log')


    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, binned_x, binned_y, name, eyeGrid) in enumerate(train_loader):
        
        with open(log_file_path, "a") as logfile:
            
            
    
        
            # measure data loading time
            data_time.update(time.time() - end)
            
            
            # imFace = imFace.cuda()
            # imEyeL = imEyeL.cuda()
            # imEyeR = imEyeR.cuda()
            # faceGrid = faceGrid.cuda()
            # gaze = gaze.cuda()
            # binned_x = binned_x.cuda()
            # binned_y = binned_y.cuda()
            
            imFace = imFace.cuda(gpu)
            imEyeL = imEyeL.cuda(gpu)
            imEyeR = imEyeR.cuda(gpu)
            faceGrid = faceGrid.cuda(gpu)
            eyeGrid = eyeGrid.cuda(gpu)
            gaze = gaze.cuda(gpu)
            binned_x = binned_x.cuda(gpu)
            binned_y = binned_y.cuda(gpu)
            
            imFace = torch.autograd.Variable(imFace, requires_grad = True)
            imEyeL = torch.autograd.Variable(imEyeL, requires_grad = True)
            imEyeR = torch.autograd.Variable(imEyeR, requires_grad = True)
            faceGrid = torch.autograd.Variable(faceGrid, requires_grad = True)
            eyeGrid = torch.autograd.Variable(eyeGrid, requires_grad = True)
            gaze = torch.autograd.Variable(gaze, requires_grad = False)
            binned_x = torch.autograd.Variable(binned_x, requires_grad = False)
            binned_y = torch.autograd.Variable(binned_y, requires_grad = False)

            ## output(binned)
            x, y , prex1= model(imFace, imEyeL, imEyeR, faceGrid, eyeGrid)
            
            # if i%100 == 0:
            #     print(f'x:{x}')
            
            
            ## CLE(Cross Entropy Loss)
            CLE_loss_x = criterion_CLE(x, binned_x)
            CLE_loss_y = criterion_CLE(y, binned_y)
            ##
            
            
            ## L2損失　(MSEloss)
            #softmax
            x_norm = softmax(x)
            y_norm = softmax(y)
            
            #binの期待値(expected value) batchサイズ分の配列になる
            x_exp = torch.sum(x_norm * idx_tensor_x, 1) * binwidth_x - x_max
            y_exp = torch.sum(y_norm * idx_tensor_y, 1) * binwidth_y
            
            #MSE
            L2_loss_x = criterion_MSE(x_exp, gaze[:,0])
            L2_loss_y = criterion_MSE(y_exp, gaze[:,1])
            ##
            
            #Total loss
            Total_loss_x = alpha * CLE_loss_x + L2_loss_x
            Total_loss_y = alpha * CLE_loss_y + L2_loss_y
            Total_losses_x.update(Total_loss_x.data.item(), imFace.size(0))
            Total_losses_y.update(Total_loss_y.data.item(), imFace.size(1))
            
            loss_seq = [Total_loss_x, Total_loss_y]
            # if i % 10 == 0:
            #     print(loss_seq)
            # grad_seq = [torch.tensor(1.0).cuda() for _ in range(len(loss_seq))]
            grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
            
            

            # 最適化
            optimizer.zero_grad()
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
            
            logger = f'{formatted_now} (train): fold[{fold}] epoch[{epoch+1}/{epochs}] iter[{i+1}/{len(train_loader)}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tLoss x:{Total_losses_x.val:.4f} ({Total_losses_x.avg:.4f})\ty:{Total_losses_y.val:.4f} ({Total_losses_y.avg:.4f})'
            logfile.write(logger+'\n')
            print(logger)
            
            
            # 損失のプロット
            if (i + (epoch) * len(train_loader)) // 50:
                pltiterlist.append(i + (epoch) * len(train_loader))
                plt.clf()
                pltlossx = Total_loss_x
                pltlossx.cpu()
                pltlossx = float(pltlossx)
                pltlossxlist.append(pltlossx)
                plt.subplot(2, 1, 1)  # 2行1列の1番目のサブプロット
                plt.plot(pltiterlist, pltlossxlist, label='Loss')

                # グラフにタイトルと軸ラベルを追加
                plt.title('loss_x')
                plt.xlabel('Total Iteration')
                plt.ylabel('Loss')

                # 凡例を表示
                plt.legend()
                
                pltlossy = Total_loss_y
                pltlossy.cpu()
                pltlossylist.append(float(pltlossy))
                plt.subplot(2, 1, 2)  # 2行1列の2番目のサブプロット
                plt.plot(pltiterlist, pltlossylist, label='Loss')
                # グラフにタイトルと軸ラベルを追加
                plt.title('loss_y')
                plt.xlabel('Total Iteration')
                plt.ylabel('Loss')
                
                # グラフを保存
                plt.savefig(os.path.join(log_path,'loss_plt.png'))


def validate(val_loader, model, criterion_CLE, criterion_MSE, softmax, idx_tensor_x, idx_tensor_y, epoch, fold, log_file_path):
    batch_time = AverageMeter()
    Total_losses_x = AverageMeter()
    Total_losses_y = AverageMeter()
    Euc_losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()


    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, binned_x, binned_y, name, eyeGrid) in enumerate(val_loader):
        try:
            with open(log_file_path, "a") as logfile:
                if epoch == 0 and i == 0:
                    logfile.write(f'PC:{pc_name}\n')
                
                # measure data loading time
                # imFace = imFace.cuda()
                # imEyeL = imEyeL.cuda()
                # imEyeR = imEyeR.cuda()
                # faceGrid = faceGrid.cuda()
                # gaze = gaze.cuda()
                # binned_x = binned_x.cuda()
                # binned_y = binned_y.cuda()
                
                imFace = imFace.cuda(gpu)
                imEyeL = imEyeL.cuda(gpu)
                imEyeR = imEyeR.cuda(gpu)
                faceGrid = faceGrid.cuda(gpu)
                eyeGrid = eyeGrid.cuda(gpu)
                gaze = gaze.cuda(gpu)
                binned_x = binned_x.cuda(gpu)
                binned_y = binned_y.cuda(gpu)
                
                imFace = torch.autograd.Variable(imFace, requires_grad = False)
                imEyeL = torch.autograd.Variable(imEyeL, requires_grad = False)
                imEyeR = torch.autograd.Variable(imEyeR, requires_grad = False)
                faceGrid = torch.autograd.Variable(faceGrid, requires_grad = False)
                eyeGrid = torch.autograd.Variable(eyeGrid, requires_grad = False)
                gaze = torch.autograd.Variable(gaze, requires_grad = False)
                binned_x = torch.autograd.Variable(binned_x, requires_grad = False)
                binned_y = torch.autograd.Variable(binned_y, requires_grad = False)

                # compute output
                with torch.no_grad():
                    x, y ,prex1= model(imFace, imEyeL, imEyeR, faceGrid, eyeGrid)
                    
                ## CLE(Cross Entropy Loss)
                CLE_loss_x = criterion_CLE(x, binned_x)
                CLE_loss_y = criterion_CLE(y, binned_y)
                ##
                
                
                ## L2損失　(MSEloss)
                #softmax
                x_norm = softmax(x)
                y_norm = softmax(y)
                # logfile.write(f'{x_norm}\n')
                
                #binの期待値(expected value) batchサイズ分の配列になる
                x_exp = torch.sum(x_norm * idx_tensor_x, 1) * binwidth_x - x_max
                y_exp = torch.sum(y_norm * idx_tensor_y, 1) * binwidth_y
                
                #MSE
                L2_loss_x = criterion_MSE(x_exp, gaze[:,0])
                L2_loss_y = criterion_MSE(y_exp, gaze[:,1])
                ##
                
                # print(CLE_loss_x)
                # print(L2_loss_x)
                
                #Total loss
                Total_loss_x = alpha * CLE_loss_x + L2_loss_x
                Total_loss_y = alpha * CLE_loss_y + L2_loss_y
                Total_losses_x.update(Total_loss_x.data.item(), imFace.size(0))
                Total_losses_y.update(Total_loss_y.data.item(), imFace.size(1))
                
                
                ## ユークリッド誤差
                # 連続推定値
                # pre_x = softmax(x)
                # pre_y = softmax(y)
                # pre_x = torch.sum(pre_x* idx_tensor_x, 1) * binwidth_x - x_max
                # pre_y = torch.sum(pre_y * idx_tensor_y, 1) * binwidth_y
                
                output = torch.cat((x_exp.unsqueeze(1), y_exp.unsqueeze(1)), 1)
                # print(f'output:{output.shape}, gaze:{gaze.shape}')
                
             
                
                # ユークリッド誤差の計算
                error = output - gaze
                # print(error[:,0])
                # print(indices)
                # print(torch.mean(Euc_loss, 0))
                Euc_loss = torch.mul(error,error)
                Euc_loss = torch.sum(Euc_loss,1)
                Euc_loss = torch.mean(torch.sqrt(Euc_loss))

                Euc_losses.update(Euc_loss.item(), imFace.size(0))
                ##
            
                # compute gradient and do SGD step
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                now = datetime.now()
                formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

                logger = f'{formatted_now} (val): fold[{fold}] epoch[{epoch+1}] iter[{i+1}/{len(val_loader)}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tLoss x:{Total_losses_x.val:.4f} ({Total_losses_x.avg:.4f})\ty:{Total_losses_y.val:.4f} ({Total_losses_y.avg:.4f})\tユークリッド誤差 {Euc_losses.val:.4f} ({Euc_losses.avg:.4f})'
                logfile.write(logger+'\n')
                print(logger)
                
        except Exception as e:
            print(f'エラー: {e}')
            
    return Euc_losses.avg

 

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
    lr = base_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


if __name__ == "__main__":
    main()
    print('DONE')
