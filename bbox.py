import os
import cv2
import dlib
import numpy as np
from tqdm import tqdm
import scipy.io as sio



# 入力画像が格納されているディレクトリのパス
INPUT = "../my_L2CSNet/MPIIFaceGaze"

# 出力画像を保存するディレクトリのパス
OUTPUT = "datasets/dlib_addeye/Image"
OUTPUTL = "datasets/dlib_addeye/Label"



def main():
    
    for person in range(15):
        my_dict = {}
        _input_directory = os.path.join(INPUT, f'p{person:02d}')
        output_directory = os.path.join(OUTPUT, f'p{person:02d}')
        day_list = os.listdir(_input_directory)
        day_list.sort()
        day_list.pop()
        day_list.pop(0)
        print(day_list)
        for day_num, day in enumerate(day_list):
            input_directory = os.path.join(_input_directory, day)
            print(input_directory)
            bbox_dlib(input_directory, output_directory, person, day_num + 1, my_dict)
            # bbox_cv(input_directory, output_directory, person, day_num + 1, my_dict)
        makemeta(person, my_dict)
        
    
    
def makemeta(person, my_dict):
    # .labelファイルのパス
    input_label_file_path = f"../my_L2CSNet/MPIIFaceGaze/p{person:02d}/p{person:02d}.txt"
    output_label = OUTPUTL
    
    if not os.path.exists(output_label):
        os.makedirs(output_label)
    
    
    output_label_file_path = os.path.join(output_label, f"p{person:02d}.label")
    
    
    excount = 0
    
    path_end = f'p{person:02d}/Calibration'
    calib_path = os.path.join("../my_L2CSNet/MPIIGaze/Data/Original", path_end)
    monitor = sio.loadmat(os.path.join(calib_path,"monitorPose.mat"))
    screen = sio.loadmat(os.path.join(calib_path,"screenSize.mat"))
    
    tvecs = monitor["tvecs"]
    x_d  = tvecs[0,0]
    y_d  = tvecs[1,0]
    height_mm = screen["height_mm"]
    height_mm = height_mm[0,0]
    height_pixel = screen["height_pixel"]
    height_pixel = height_pixel[0,0]
    width_mm = screen["width_mm"]
    width_mm = width_mm[0,0]
    width_pixel = screen["width_pixel"]
    width_pixel = width_pixel[0,0]
    
    ###
    x_d = width_mm / 2
    y_d = -1
    
    

    # .labelファイルを開いて読み込み
    with open(input_label_file_path, 'r') as input_label_file, open(output_label_file_path, 'w') as output_label_file:
        for line in input_label_file:
            
            try:
                # 各行をスペースで分割
                elements = line.strip().split()
                sgazex_pix  = int(elements[1])
                sgazey_pix  = int(elements[2])
                
                sgazex_mm = sgazex_pix * width_mm / width_pixel
                sgazey_mm = sgazey_pix * height_mm / height_pixel
                
                gazex_mm = sgazex_mm - x_d
                gazey_mm = sgazey_mm - y_d
                
                
                
                _a = elements[0].split('/')[1]#YYYY.jpg
                _b = elements[0].split('/')[0]#dayXX
                filenum = _a.split('.')[0]#YYYY
                daynum = _b.split('y')[1]#XX
                
                labelidx  = f'p{person:02d}/' + daynum + filenum #pZZ/XXYYYY
                
                if labelidx == "p12/010036":
                    print('a')
                
                bbox = my_dict[labelidx]
                if labelidx == "p12/010036":
                    print('b')
                # x = bbox[0]
                # y = bbox[1]
                # w = bbox[2]
                # h = bbox[3]
                x, y, w, h, lx, ly, lw, lh, rx, ry, rw, rh = bbox
                
                # 1.2.3番目の要素だけを取得
                selected_elements = [labelidx, str(gazex_mm), str(gazey_mm), str(x), str(y), str(w), str(h), str(lx), str(ly), str(lw), str(lh), str(rx), str(ry), str(rw), str(rh)]
                
                # 新しい.labelファイルに書き込み
                output_label_file.write(" ".join(selected_elements) + "\n")
            except Exception as e:
                excount += 1
                print(f"エラー: {e}, 画像 {elements[0]} を処理できませんでした。")
                
        print(f'excount{excount}')
            
    print(f"Data saved to {output_label_file_path}")


def bbox_cv(input_directory, output_directory, person, day, my_dict):
    # 入力画像が格納されているディレクトリのパス
    # input_directory = "/workspace-cloud/koki.murata/my_L2CSNet/MPIIFaceGaze/p00/day01"

    # 出力画像を保存するディレクトリのパス
    # output_directory_face = "/workspace-cloud/koki.murata/MPIITracker/datasets/Image/p00/face"
    # output_directory_left_eye = "/workspace-cloud/koki.murata/MPIITracker/datasets/Image/p00/left"
    # output_directory_right_eye = "/workspace-cloud/koki.murata/MPIITracker/datasets/Image/p00/right"
    # output_directory_mask = "/workspace-cloud/koki.murata/MPIITracker/datasets/Image/p00/mask"
    
    output_directory_face = os.path.join(output_directory, 'face')
    output_directory_left_eye = os.path.join(output_directory, 'left')
    output_directory_right_eye = os.path.join(output_directory, 'right')
    output_directory_mask = os.path.join(output_directory, 'mask')

    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_directory_face):
        os.makedirs(output_directory_face)

    if not os.path.exists(output_directory_left_eye):
        os.makedirs(output_directory_left_eye)

    if not os.path.exists(output_directory_right_eye):
        os.makedirs(output_directory_right_eye)
        
    
    if not os.path.exists(output_directory_mask):
        os.makedirs(output_directory_mask)

    # 顔検出用のHaar Cascade分類器の読み込み
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 左目検出用のHaar Cascade分類器の読み込み
    left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')

    # 右目検出用のHaar Cascade分類器の読み込み
    right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
    
    # 新しい画像のサイズ
    new_size = (224,224)
    new_mask_size = (25, 25)

    # 入力ディレクトリ内のすべての画像ファイルに対して処理を行う
    for filename in tqdm(os.listdir(input_directory), desc=f"Processing images/person{person:02d}/day{day:02d}"):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # 画像ファイルの拡張子を指定
            try:
                # 画像の読み込み
                image_path = os.path.join(input_directory, filename)
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                filenum_str = filename.split('.')[0]#YYYY
                dayfilenum = f"{day:02d}{filenum_str}"#XXYYYY
                
                
                
                # 顔の検出
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                

                if not isinstance(faces, np.ndarray):
                    raise TypeError("顔検出できませんでした。")

                # 各検出された顔に対して処理を行う
                for (x, y, w, h) in faces:
                    
                    
                    # 顔領域を切り取る
                    face_roi = image[y:y + h, x:x + w]
                    
                    #リサイズ
                    resized_face = cv2.resize(face_roi, new_size, interpolation=cv2.INTER_AREA)


                    # 出力ファイルのパス
                    output_filename_face = f"{dayfilenum}_face.jpg"
                    output_path_face = os.path.join(output_directory_face, output_filename_face)

                    # 顔の画像を保存
                    cv2.imwrite(output_path_face, resized_face)
                    
                    # マスク画像の作成
                    mask = np.ones_like(image) * 255  # 全体が白い画像
                    mask[y:y + h, x:x + w] = 0  # 顔の領域を黒くする
                    
                    #マスク画像をリサイズ
                    resized_mask = cv2.resize(mask, new_mask_size, interpolation=cv2.INTER_AREA)

                    # マスク画像の出力ファイルのパス
                    output_filename_mask = f"{dayfilenum}_mask.jpg"
                    output_path_mask = os.path.join(output_directory_mask, output_filename_mask)

                    # マスク画像を保存
                    cv2.imwrite(output_path_mask, resized_mask)
                    
                    

                    # 顔領域内で左目を検出
                    left_eyes = left_eye_cascade.detectMultiScale(gray[y:y + h, x:x + w])
                    print(left_eyes)
                    
                    if dayfilenum == '080661':
                        print(f'080661{left_eyes}')
                    
                    if not isinstance(left_eyes, np.ndarray):
                        raise TypeError("左目検出できませんでした。")
                    

                    # 各検出された左目に対して処理を行う
                    for (ex, ey, ew, eh) in left_eyes:
                        # 左目領域を切り取る
                        left_eye_roi = face_roi[ey:ey + eh, ex:ex + ew]
                        
                        # リサイズ
                        resized_left_eye = cv2.resize(left_eye_roi, new_size, interpolation=cv2.INTER_AREA)

                        # 出力ファイルのパス
                        output_filename_left_eye = f"{dayfilenum}_left.jpg"
                        output_path_left_eye = os.path.join(output_directory_left_eye, output_filename_left_eye)

                        # 左目の画像を保存
                        cv2.imwrite(output_path_left_eye, resized_left_eye)

                    # 顔領域内で右目を検出
                    right_eyes = right_eye_cascade.detectMultiScale(gray[y:y + h, x:x + w])
                    print(right_eyes)
                    
                    if not isinstance(right_eyes, np.ndarray):
                        raise TypeError("右目検出できませんでした。")

                    # 各検出された右目に対して処理を行う
                    for (ex, ey, ew, eh) in right_eyes:
                        # 右目領域を切り取る
                        right_eye_roi = face_roi[ey:ey + eh, ex:ex + ew]
                        
                        # リサイズ
                        resized_right_eye = cv2.resize(right_eye_roi, new_size, interpolation=cv2.INTER_AREA)

                        # 出力ファイルのパス
                        output_filename_right_eye = f"{dayfilenum}_right.jpg"
                        output_path_right_eye = os.path.join(output_directory_right_eye, output_filename_right_eye)

                        # 右目の画像を保存
                        cv2.imwrite(output_path_right_eye, resized_right_eye)
                        
                    #bboxの点を辞書に保存
                    dict_key = f'p{person:02d}/' + dayfilenum #pZZ/XXYYYY
                    my_dict[dict_key] = (x, y, w, h)


            except Exception as e:
                print(f"エラー: {e}, 画像 {dayfilenum} を処理できませんでした。")
                
            except TypeError as e:
                print(f"エラー: {e}, 画像: {dayfilenum} ")

    print("処理が完了しました。")
    
    
    
def bbox_dlib(input_directory, output_directory, person, day, my_dict):
    # 入力画像が格納されているディレクトリのパス
    # input_directory = "/workspace-cloud/koki.murata/my_L2CSNet/MPIIFaceGaze/p00/day01"

    # 出力画像を保存するディレクトリのパス
    # output_directory_face = "/workspace-cloud/koki.murata/MPIITracker/datasets/Image/p00/face"
    # output_directory_left_eye = "/workspace-cloud/koki.murata/MPIITracker/datasets/Image/p00/left"
    # output_directory_right_eye = "/workspace-cloud/koki.murata/MPIITracker/datasets/Image/p00/right"
    # output_directory_mask = "/workspace-cloud/koki.murata/MPIITracker/datasets/Image/p00/mask"
    
    output_directory_face = os.path.join(output_directory, 'face')
    output_directory_left_eye = os.path.join(output_directory, 'left')
    output_directory_right_eye = os.path.join(output_directory, 'right')
    output_directory_mask = os.path.join(output_directory, 'mask')

    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_directory_face):
        os.makedirs(output_directory_face)

    if not os.path.exists(output_directory_left_eye):
        os.makedirs(output_directory_left_eye)

    if not os.path.exists(output_directory_right_eye):
        os.makedirs(output_directory_right_eye)
        
    
    if not os.path.exists(output_directory_mask):
        os.makedirs(output_directory_mask)

   # dlibの顔検出器とランドマーク検出器を初期化
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("dlib_model/shape_predictor_68_face_landmarks.dat")
    
    # 新しい画像のサイズ
    new_size = (224,224)
    new_mask_size = (25, 25)

    # 入力ディレクトリ内のすべての画像ファイルに対して処理を行う
    for filename in tqdm(os.listdir(input_directory), desc=f"Processing images/person{person:02d}/day{day:02d}"):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # 画像ファイルの拡張子を指定
            try:
                # 画像の読み込み
                image_path = os.path.join(input_directory, filename)
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                filenum_str = filename.split('.')[0]#YYYY
                dayfilenum = f"{day:02d}{filenum_str}"#XXYYYY
                
                
                
                # 顔の検出
                faces = detector(gray)
                # print(faces)
                

                # if isinstance(faces, np.ndarray):
                #     raise TypeError("顔検出できませんでした。")

                # 各検出された顔に対して処理を行う
                for face in faces:
                    # 顔領域を切り取る
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    face_roi = image[y:y + h, x:x + w]
                    
                    
                    #リサイズ
                    resized_face = cv2.resize(face_roi, new_size, interpolation=cv2.INTER_AREA)


                    # 出力ファイルのパス
                    output_filename_face = f"{dayfilenum}_face.jpg"
                    output_path_face = os.path.join(output_directory_face, output_filename_face)

                    # 顔の画像を保存
                    cv2.imwrite(output_path_face, resized_face)
                    
                    # マスク画像の作成
                    mask = np.ones_like(image) * 255  # 全体が白い画像
                    mask[y:y + h, x:x + w] = 0  # 顔の領域を黒くする
                    
                    #マスク画像をリサイズ
                    resized_mask = cv2.resize(mask, new_mask_size, interpolation=cv2.INTER_AREA)

                    # マスク画像の出力ファイルのパス
                    output_filename_mask = f"{dayfilenum}_mask.jpg"
                    output_path_mask = os.path.join(output_directory_mask, output_filename_mask)

                    # マスク画像を保存
                    cv2.imwrite(output_path_mask, resized_mask)
                    
                    

                    # ランドマークの検出
                    shape = predictor(gray, face)
                    
                    
                    # if not isinstance(shape, np.ndarray):
                        # raise TypeError("左目検出できませんでした。")
                    
                    # 左目の検出
                    lcenter_x = (shape.part(36).x + shape.part(39).x) // 2
                    lcenter_y = (shape.part(36).y + shape.part(39).y) // 2
                    left_eye_roi = image[lcenter_y-40:lcenter_y+40, lcenter_x-40:lcenter_x+40]
                    # print(shape.part(36))
                    # print(shape.part(37))
                    # print(shape.part(38))
                    # print(shape.part(39))
                    # print(shape.part(40))
                    # print(shape.part(41))
                    # print(shape.part(42))
                    # print(shape.part(43))
                    # print(shape.part(44))
                    # print(shape.part(45))
                    # print(shape.part(46))
                    # print(shape.part(47))
                    # print(left_eye_roi)
                    resized_left_eye = cv2.resize(left_eye_roi, new_size, interpolation=cv2.INTER_AREA)
                    lx = shape.part(36).x
                    ly = (shape.part(37).y + shape.part(38).y) // 2
                    lw = shape.part(39).x - lx
                    lh = (shape.part(40).y + shape.part(41).y) // 2 - ly

                    # 出力ファイルのパス
                    output_filename_left_eye = f"{dayfilenum}_left.jpg"
                    output_path_left_eye = os.path.join(output_directory_left_eye, output_filename_left_eye)

                    # 左目の画像を保存
                    cv2.imwrite(output_path_left_eye, resized_left_eye)
                    # cv2.imwrite(output_path_left_eye, left_eye_roi)
                    
                    
                    # 右目の検出
                    rcenter_x = (shape.part(42).x + shape.part(45).x) // 2
                    rcenter_y = (shape.part(42).y + shape.part(45).y) // 2
                    right_eye_roi = image[rcenter_y-40:rcenter_y+40, rcenter_x-40:rcenter_x+40]
                    resized_right_eye = cv2.resize(right_eye_roi, new_size, interpolation=cv2.INTER_AREA)
                    rx = shape.part(42).x
                    ry = (shape.part(43).y + shape.part(44).y) // 2
                    rw = shape.part(45).x - lx
                    rh = (shape.part(46).y + shape.part(47).y) // 2 - ly
                        

                    # 出力ファイルのパス
                    output_filename_right_eye = f"{dayfilenum}_right.jpg"
                    output_path_right_eye = os.path.join(output_directory_right_eye, output_filename_right_eye)

                    # 右目の画像を保存
                    cv2.imwrite(output_path_right_eye, resized_right_eye)
                        
                    #bboxの点を辞書に保存
                    dict_key = f'p{person:02d}/' + dayfilenum #pZZ/XXYYYY
                    my_dict[dict_key] = (x, y, w, h, lx, ly, lw, lh, rx, ry, rw, rh)


            except Exception as e:
                print(f"エラー: {e}, 画像 {dayfilenum} を処理できませんでした。")
                
            except TypeError as e:
                print(f"エラー: {e}, 画像: {dayfilenum} ")

    print("処理が完了しました。")
    


if __name__ == "__main__":
    main()
    print('DONE')