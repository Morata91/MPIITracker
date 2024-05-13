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


from ITrackerData import ITrackerData
from ITrackerModel import ITrackerModel


batch_size = 100

CP_PATH = 'checkpoints/202311280416'




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
    
    labelpathlist = os.listdir("datasets/Label")
    labelpathlist.sort()
    impath = "datasets/Image"
    cp_fold_list = os.listdir(CP_PATH)
    cp_fold_list.sort()
    print(cp_fold_list)

    # for fold in range(0,7):
    for fold in range(len(cp_fold_list)):
        test_data = ITrackerData(labelpathlist, impath, fold=fold, train=False)
    
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
        folder = os.listdir(checkpoint_path)
        folder.sort()#01,02,03,...,best
        print(folder)
        
        
        
        print(evallogpath)
        
        with open(os.path.join(evallogpath, 'eval.log'), 'w') as outfile:
            configuration = f"\ntest configuration equal gpu_id=?, batch_size={batch_size}, fold={fold}---------------------------------------\n"
            print(configuration)
            outfile.write(configuration)
            epoch_list=[]
            avg_MAE=[]
            model = ITrackerModel()
            model.cuda()
            for epochs in folder: 
                if onlybest:
                    if not epochs == 'best.pth.tar':
                        continue
                else:
                    if epochs == 'best.pth.tar':
                        continue
                    
                # try:
                    
                # checkpoint_path=YYMMDDHHMM/fold_XX
                # YY.pth.tar
                # cp_time, fold_saved, epoch_saved = checkpoint_path.split('/')
                _cp = checkpoint_path.split('/')
                fold_saved = _cp[6].split('_')[1]   
                starting_fold = int(fold_saved)
                saved = torch.load(os.path.join(checkpoint_path, epochs))
                saved_state_dict = saved['state_dict']
                    
                    
                # except Exception as e:
                #     print(f'エラー: {e}, checkpointが読み込めませんでした.')
                #     sys.exit(1)
                # saved_state_dict = torch.load(os.path.join(snapshot_path+"/fold"+str(fold),epochs))
                # model= nn.DataParallel(model,device_ids=[0])
                model.load_state_dict(saved_state_dict)
                model.cuda()
                log_file_path = os.path.join(evallogpath, 'log.log')
                evaluation(test_loader, model, log_file_path, fold)
    
        # fig = plt.figure(figsize=(14, 8))        
        # plt.xlabel('epoch')
        # plt.ylabel('avg')
        # plt.title('Gaze angular error')
        # plt.legend()
        # plt.plot(epoch_list, avg_MAE, color='k', label='mae')
        if not os.path.exists(os.path.join(evalpath,"fold"+str(fold))):
            os.makedirs(os.path.join(evalpath,"fold"+str(fold)))
        # fig.savefig(os.path.join(evalpath, os.path.join("fold"+str(fold), "best.png")), format='png')#
        # plt.show()

            
def evaluation(test_loader, model, log_file_path, fold):
    global count_test
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    
    criterion = nn.MSELoss().cuda()


    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze) in enumerate(test_loader):
        try:
            with open(log_file_path, "a") as logfile:
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

                logger = f'{formatted_now} (test): fold[{fold}] iter[{i+1}/{len(test_loader)}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tLoss {losses.val:.4f} ({losses.avg:.4f})\tError L2 {lossesLin.val:.4f} ({lossesLin.avg:.4f})'
                logfile.write(logger+'\n')
                print(logger)
        except Exception as e:
            print(f'エラー: {e}')



    return lossesLin.avg
           
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
        
        
if __name__ == "__main__":
    main()
    print('DONE')