import os
import numpy as np








def main():
    path = '/workspace-cloud/koki.murata/MPIITracker/datasets/dlib1/Label'

    pathlist = os.listdir(path)
    x_max=150
    y_max=200
    binwidth_x = 10
    binwidth_y = 10
    xbins_num = x_max*2//binwidth_x
    ybins_num = y_max//binwidth_y
    print(f'ビン数・・・x:{xbins_num}  y:{ybins_num}')
    counterx = [0] * xbins_num
    countery = [0] * ybins_num
    orig_list_len = 0
    for i in pathlist:
        with open(os.path.join(path,i)) as f:
            lines = f.readlines()
            lines.pop(0)
            print('lines:{}'.format(len(lines)))
            orig_list_len += len(lines)
            for line in lines:
                x = line.strip().split(" ")[1]
                y = line.strip().split(" ")[2]
                
                x = float(x)
                y = float(y)
                
                #binの作成
                bins_x = np.array(range(-1*x_max, x_max, binwidth_x))
                bins_y = np.array(range(0, y_max, binwidth_y))
                
                #binのインデックスの取得
                binned_x = np.digitize(x, bins_x) - 1
                binned_y = np.digitize(y, bins_y) - 1
                
                
                counterx[binned_x] += 1
                countery[binned_y] += 1
                        
    print(counterx)
    print(countery)
    
    
def main1():
    path = '/workspace-cloud/koki.murata/my_L2CSNet/MPIIFaceGaze/p00/p00.txt'
    x_max=1300
    y_max=1000
    binwidth_x = 50
    binwidth_y = 50
    xbins_num = x_max//binwidth_x
    ybins_num = y_max//binwidth_y
    print(f'ビン数・・・x:{xbins_num}  y:{ybins_num}')
    # counter = np.zeros(xbins_num, dtype=int)
    counter = [0] * xbins_num
    orig_list_len = 0
    with open(path) as f:
        lines = f.readlines()
        lines.pop(0)
        print('lines:{}'.format(len(lines)))
        orig_list_len += len(lines)
        #binの作成
        bins_x = np.array(range(0, x_max, binwidth_x))
        bins_y = np.array(range(0, y_max, binwidth_y))
        print(bins_x)
        for line in lines:
            x = line.strip().split(" ")[1]
            y = line.strip().split(" ")[2]
            
            x = int(x)
            
            
            #binのインデックスの取得
            binned_x = np.digitize(x, bins_x) - 1
            binned_y = np.digitize(y, bins_y) - 1
            
            
            # for i in range(len(counter)):
            #     if i == binned_x:
            counter[binned_x] += 1
                        
    print(counter)
            
            
            
if __name__ == '__main__':
    main()
            