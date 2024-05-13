

import argparse
import pathlib
import numpy as np
import cv2
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

import os
import ffmpeg
import subprocess as sp

from PIL import Image
from PIL import Image, ImageOps

import dlib

from pipeline import Pipeline
from vis import render


CWD = pathlib.Path.cwd()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='.')
    parser.add_argument(
        '--device',dest='device', help='Device to run model: cpu or gpu:0',
        default="cpu", type=str)
    parser.add_argument(
        '--cp', help='Path of model snapshot.', 
        default='none', type=str)
    parser.add_argument(
        '--datatype',dest='datatype', help='カメラ:0, 動画(.mov):1, 画像:2',  
        default=0, type=int)
    parser.add_argument(
        '--data',dest='data',help='使用データ.mov(.jpg)',
        default='video.mov', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    datatype = args.datatype
    data = args.data
    snapshot_path = args.cp


    gaze_pipeline = Pipeline(
        weights=CWD / snapshot_path,
    )
    
    if datatype == 0:
        cap = cv2.VideoCapture(0)
        
        
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        else:
            print("カメラキャプチャします")

        with torch.no_grad():
                while True:

                # Get frame
                    success, frame = cap.read()    
                    start_fps = time.time()  

                    if not success:
                        print("Failed to obtain frame")
                        time.sleep(0.1)

                    # Process frame
                    results = gaze_pipeline.step(frame)

                    # Visualize output
                    frame = render(frame, results)
                
                    myFPS = 1.0 / (time.time() - start_fps)
                    cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

                    cv2.imshow("Demo",frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    success,frame = cap.read()  
        
        
    elif datatype == 1: #動画
        input_file = data
        file_name = os.path.splitext(os.path.basename(input_file))[0]
        use_file = file_name + '.mp4'
        
        if not os.path.isfile(use_file):

            cmd_list = ['ffmpeg', '-i', input_file, file_name + '.mp4']
            cmd = ' '.join(cmd_list)
            sp.call(cmd, shell=True)
            print(input_file + 'を' + use_file + 'に変換しました')
        else:
            print(use_file + 'がすでに存在したため変換は行いませんでした')
            
        cap = cv2.VideoCapture(use_file)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # フレームの幅
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # フレームの高さ
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # FPS
        # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # FOURCC
        # fourcc = fourcc.to_bytes(4, "little").decode("utf-8")  # int を4バイトずつ解釈する。
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # フレーム数の合計
        TEMP_VIDEO = 'temp.mjpg'
        OUT_VIDEO = 'result.mp4'
        

        print(f"width: {width}\n height: {height}\n fps: {fps}\n forcc: {fourcc}\n total frames: {n_frames}\n")
        play_time = n_frames / fps
        print('{} frames, {:1.1f} (s)'.format(n_frames, play_time))

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam or video")
        
        pbar = tqdm(total=n_frames)
        
        logger_path = "result.log"

        with torch.no_grad():
            # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (width, height))
            
            
            with open(logger_path, 'w') as outfile:
            
                while True:
                    

                    # Get frame
                    success, frame = cap.read()    
                    
                    start_fps = time.time()  

                    if not success:
                        print("Failed to obtain frame")
                        time.sleep(0.1)
                        break
                        

                    # Process frame
                    # class GazeResultContainer:
                    #     x: np.ndarray
                    #     y: np.ndarray
                    #     bboxes: np.ndarray
                    #     landmarks: np.ndarray
                    #     scores: np.ndarray
                    
                    results = gaze_pipeline.step(frame, width, height)
                    
                    logger = f"x:{results.x},y:{results.y}"
                    outfile.write(logger+'\n')

                    # Visualize output
                    # frame = render(frame, results)
                
                    # myFPS = 1.0 / (time.time() - start_fps)
                    # cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

                    
                    #cv2.imshow("Demo",frame)

                    
                    # writer.write(frame)
                    pbar.update(1)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    # success,frame = cap.read()  
        
        pbar.close()
        # ffmpeg.input(TEMP_VIDEO).output('sample/result.mp4', vcodec='libx264').run(overwrite_output=True)
        # os.remove(TEMP_VIDEO)


    elif datatype == 2: #画像
        OUT_PHOTO = "sample/result.jpg"
        input_file = data
        data_f = os.path.splitext(os.path.basename(input_file))[1]
        print(data_f)
        assert data_f == ".jpg"
            
        cap = cv2.imread(input_file)
        
        width = cap.shape[1]  # フレームの幅
        height = cap.shape[0]  # フレームの高さ

        print(f"width: {width}\n height: {height}\n")

        
        
        logger_path = "sample/result.log"

        with torch.no_grad():
            with open(logger_path, 'w') as outfile:
            
                # Get frame
                frame = cap    
                    

                # Process frame
                # class GazeResultContainer:
                #     x: np.ndarray
                #     y: np.ndarray
                #     bboxes: np.ndarray
                #     landmarks: np.ndarray
                #     scores: np.ndarray
                
                results = gaze_pipeline.step(frame)
                
                logger = f"x:{results.x},y:{results.y},landmarks{results.landmarks}"
                outfile.write(logger)

                # Visualize output
                frame = render(frame, results)
                
                cv2.imwrite(OUT_PHOTO, frame)
            
                # myFPS = 1.0 / (time.time() - start_fps)
                # cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

                
                #cv2.imshow("Demo",frame)

                    
        
        # ffmpeg.input(TEMP_VIDEO).output('sample/result.mp4', vcodec='libx264').run(overwrite_output=True)
        # os.remove(TEMP_VIDEO)