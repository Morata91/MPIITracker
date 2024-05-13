import os
import cv2
import numpy as np
from tqdm import tqdm


def main():
    image_path = "/workspace-cloud/koki.murata/MPIITracker/datasets/Image/p00/mask/0001_mask.jpg"
    image = cv2.imread(image_path)
    print(image)




if __name__ == "__main__":
    main()
    print('DONE')
    
    
    