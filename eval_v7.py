'''
train_v7での学習モデル用test.py
'''


import os, argparse, time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

import sys
from datetime import datetime


from MPIITrackerData_v1 import MPIITrackerData
from MPIITrackerModel_v4 import MPIITrackerModel


batch_size = 15
##binに関する変数の指定
x_max=150
y_max=200
binwidth_x = 30
binwidth_y = 20


CP_PATH = 'checkpoints/240118_1'

pltalllist = np.zeros([15,50])
pltalllist[:,:] = np.nan
# pltalllist = np.zeros((15,50))
AVGPATH = 'evallog/avg'


pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'




def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Gaze estimation using L2CSNet .')
    parser.add_argument('--onlybest', type=str2bool, nargs='?', const=True, default=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    onlybest = args.onlybest
    cudnn.enabled = True
    # bins=args.bins
    # bins = 28
    # angle=args.angle
    # bin_width=args.bin_width
    # model_used=getArch(arch, bins)
    
    xbins_num = x_max*2//binwidth_x
    ybins_num = y_max//binwidth_y
    idx_tensor_x = [idx for idx in range(xbins_num)]
    idx_tensor_x = torch.autograd.Variable(torch.FloatTensor(idx_tensor_x)).cuda()
    idx_tensor_y = [idx for idx in range(ybins_num)]
    idx_tensor_y = torch.autograd.Variable(torch.FloatTensor(idx_tensor_y)).cuda()

    
    labelpathlist = os.listdir("datasets/Label")
    labelpathlist.sort()
    impath = "datasets/Image"
    cp_fold_list = os.listdir(CP_PATH)
    cp_fold_list.sort()
    print(cp_fold_list)

    # for fold in range(10,15):
    for fold in range(len(cp_fold_list)):
        pltepochlist = []
        plterrorlist = []
        test_data = MPIITrackerData(labelpathlist, impath, fold=fold, train=False, x_max=x_max, y_max=y_max, binwidth_x=binwidth_x, binwidth_y=binwidth_y)
    
        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True)
        
        #evalationのlogを書くパスを作成
        evallogpath = os.path.join('evallog', f"fold_"+f'{fold:02d}')
        if not os.path.exists(evallogpath):
            os.makedirs(evallogpath)

        # list all epochs for testing
        #以下編集
        checkpoint_path = os.path.join(CP_PATH,cp_fold_list[fold])
        # checkpoint_path = os.path.join(CP_PATH,cp_fold_list[0])
        print(checkpoint_path)
        folder = os.listdir(checkpoint_path)
        folder.sort()#01,02,03,...,best
        print(folder)
        
        
        
        print(evallogpath)
        
        with open(os.path.join(evallogpath, 'eval.log'), 'a') as outfile:
            configuration = f"\ntest configuration equal gpu_id=?, batch_size={batch_size}, fold={fold}---------------------------------------\n"
            print(configuration)
            outfile.write(configuration)
            epoch_list=[]
            avg_MAE=[]
            model = MPIITrackerModel(xbins_num, ybins_num)
            model.cuda()
            for epochs in folder: 
                if onlybest:
                    if not epochs == 'best.pth.tar':
                        continue
                else:
                    if epochs == 'best.pth.tar':
                        continue


                print(f'epoch:{epochs}')
                outfile.write(f'epoch:{epochs}')
                    
                # try:
                    
                # checkpoint_path=YYMMDDHHMM/fold_XX
                # YY.pth.tar
                # cp_time, fold_saved, epoch_saved = checkpoint_path.split('/')
                _cp = checkpoint_path.split('/')
                # print(_cp)
                fold_saved = _cp[2].split('_')[1]   
                starting_fold = int(fold_saved)
                print(os.path.join(checkpoint_path, epochs))
                
                
                saved = torch.load(os.path.join(checkpoint_path, epochs))
                try:
                    print('yyy')
                    saved_state_dict = saved['state_dict']
                    model.load_state_dict(saved_state_dict)
                except:
                    continue
                
                    
                    
                # except Exception as e:
                #     print(f'エラー: {e}, checkpointが読み込めませんでした.')
                #     sys.exit(1)
                # saved_state_dict = torch.load(os.path.join(snapshot_path+"/fold"+str(fold),epochs))
                # model= nn.DataParallel(model,device_ids=[0])
                # model.load_state_dict(saved_state_dict)
                model.cuda()
                error = evaluation(test_loader, model, evallogpath, fold, idx_tensor_x, idx_tensor_y, epochs, onlybest)
                if not onlybest:
                    pltepochlist.append(int(epochs.split('.')[0]))
                    print(f'eplist{pltepochlist}')
                    plterrorlist.append(error)
                    
                    pltalllist[fold][int(epochs.split('.')[0])-1] = error
                    
                    plt.clf()
                    plt.plot(pltepochlist, plterrorlist, label='Euc Error')
                    plt.title(f'fold{fold}')
                    plt.xlabel('Total Epoch')
                    plt.ylabel('Euclid Error [mm]')
                    plt.savefig(os.path.join(evallogpath,f'{fold}'+'_loss_plt.png'))
                    
    pltavglist = np.nanmean(pltalllist, axis=0)
    if not os.path.exists(AVGPATH):
        os.makedirs(AVGPATH)
    plt.clf()
    plt.plot(pltepochlist, pltavglist, label='Euc Error')
    plt.title(f'Average')
    plt.xlabel('Total Epoch')
    plt.ylabel('Euclid Error [mm]')
    plt.savefig(os.path.join(AVGPATH,'avg_loss_plt.png'))
    
    
        # fig = plt.figure(figsize=(14, 8))        
        # plt.xlabel('epoch')
        # plt.ylabel('avg')
        # plt.title('Gaze angular error')
        # plt.legend()
        # plt.plot(epoch_list, avg_MAE, color='k', label='mae')
        # if not os.path.exists(os.path.join(evalpath,"fold"+str(fold))):
        #     os.makedirs(os.path.join(evalpath,"fold"+str(fold)))
        # fig.savefig(os.path.join(evalpath, os.path.join("fold"+str(fold), "best.png")), format='png')#
        # plt.show()

            
def evaluation(test_loader, model, evallogpath, fold, idx_tensor_x, idx_tensor_y, epochs, onlybest):
    log_file_path = os.path.join(evallogpath, 'eval.log')

    batch_time = AverageMeter()
    Total_losses_x = AverageMeter()
    Total_losses_y = AverageMeter()
    Euc_losses = AverageMeter()
    
    alpha = 1
    x_max=150
    y_max=200
    binwidth_x = 30
    binwidth_y = 20
    
    
    
    criterion_MSE = nn.MSELoss().cuda()
    criterion_CLE = nn.CrossEntropyLoss().cuda()
    softmax = nn.Softmax(dim=1).cuda()
    
    # switch to evaluate mode
    model.eval()
    end = time.time()
    


    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, binned_x, binned_y, imFacePath) in enumerate(test_loader):
        try:
            with open(log_file_path, "a") as logfile:
                # measure data loading time
                imFace = imFace.cuda()
                imEyeL = imEyeL.cuda()
                imEyeR = imEyeR.cuda()
                faceGrid = faceGrid.cuda()
                gaze = gaze.cuda()
                binned_x = binned_x.cuda()
                binned_y = binned_y.cuda()
                
                imFace = torch.autograd.Variable(imFace, requires_grad = False)
                imEyeL = torch.autograd.Variable(imEyeL, requires_grad = False)
                imEyeR = torch.autograd.Variable(imEyeR, requires_grad = False)
                faceGrid = torch.autograd.Variable(faceGrid, requires_grad = False)
                gaze = torch.autograd.Variable(gaze, requires_grad = False)
                binned_x = torch.autograd.Variable(binned_x, requires_grad = False)
                binned_y = torch.autograd.Variable(binned_y, requires_grad = False)

                # compute output
                with torch.no_grad():
                    x, y, pre_x1 = model(imFace, imEyeL, imEyeR, faceGrid)

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
                Total_loss_x = CLE_loss_x + alpha * L2_loss_x
                Total_loss_y = CLE_loss_y + alpha * L2_loss_y
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
                Euc_loss = output - gaze
                # print(Euc_loss)
                # print(torch.mean(Euc_loss, 0))
                Euc_loss = torch.mul(Euc_loss,Euc_loss)
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

                logger = f'{formatted_now} (test): fold[{fold}] iter[{i+1}/{len(test_loader)}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tLoss x:{Total_losses_x.val:.4f} ({Total_losses_x.avg:.4f})\ty:{Total_losses_y.val:.4f} ({Total_losses_y.avg:.4f})\tユークリッド誤差 {Euc_losses.val:.4f} ({Euc_losses.avg:.4f})'
                logfile.write(logger+'\n')
                print(logger)
        
        except Exception as e:
            print(f'エラー: {e}')

    return Euc_losses.avg
           
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
        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
def load_checkpoint(filename):
    filename = os.path.join(filename)
    print(f'checkpoint file:{filename}')
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state
        
        
if __name__ == "__main__":
    main()
    print('DONE')