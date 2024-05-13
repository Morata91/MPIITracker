import numpy as np
import scipy.io as sio
import cv2 
import os
import sys


#正規化なし

root = "/workspace-cloud/koki.murata/my_L2CSNet/MPIIFaceGaze"
sample_root = "/workspace-cloud/koki.murata/my_L2CSNet/MPIIGaze/Evaluation Subset/sample list for eye image"
out_root = "/workspace-cloud/koki.murata/MPIITracker/datasets"

def ImageProcessing_MPII():
    persons = os.listdir(sample_root)
    persons.sort()
    for person in persons:
        sample_list = os.path.join(sample_root, person) #/pXX.txt

        person = person.split(".")[0]#pXX
        im_root = os.path.join(root, person)
        anno_path = os.path.join(root, person, f"{person}.txt")

        im_outpath = os.path.join(out_root, person)
        label_outpath = os.path.join(out_root, person, f"{person}.label")

        if not os.path.exists(im_outpath):
            os.makedirs(im_outpath)
        if not os.path.exists(os.path.join(out_root, "Label")):
            os.makedirs(os.path.join(out_root, "Label"))

        print(f"Start Processing {person}")
        
        
        with open(anno_path) as infile:
            anno_info = infile.readlines()
        anno_dict = {line.split(" ")[0]: line.strip().split(" ")[1:-1] for line in anno_info}
        
        
        outfile = open(label_outpath, 'w')
        outfile.write("Face Left Right Origin 2DGaze\n")
        if not os.path.exists(os.path.join(im_outpath, "face")):
            os.makedirs(os.path.join(im_outpath, "face"))
        if not os.path.exists(os.path.join(im_outpath, "left")):
            os.makedirs(os.path.join(im_outpath, "left"))
        if not os.path.exists(os.path.join(im_outpath, "right")):
            os.makedirs(os.path.join(im_outpath, "right"))
            
        with open(sample_list) as infile:
            im_list = infile.readlines()
            total = len(im_list)
            
        for count, info in enumerate(im_list):
            progressbar = "".join(["\033[41m%s\033[0m" % '   '] * int(count/total * 20))
            progressbar = "\r" + progressbar + f" {count}|{total}"
            print(progressbar, end = "", flush=True)
            
            
            im_info, which_eye = info.strip().split(" ")
            day, im_name = im_info.split("/")
            im_number = int(im_name.split(".")[0]) 
            
            im_path = os.path.join(im_root, im_info)
            im = cv2.imread(im_path)
            annotation = anno_dict[im_info]
            annotation = AnnoDecode(annotation) 
            
            # im_face = GetImage(im)
            im_face = im
            
             # Crop left eye images
            llc = np.array(annotation["left_left_corner"]).astype("float32")
            lrc = np.array(annotation["left_right_corner"]).astype("float32")
            im_left = CropEye(im_face, (224, 224), llc, lrc)
            
            # Crop Right eye images
            rlc = np.array(annotation["right_left_corner"])
            rrc = np.array(annotation["right_right_corner"])
            im_right = CropEye(im_face, (224, 224), rlc, rrc)
            
            gaze = np.array(annotation["gazepx"])
            
            cv2.imwrite(os.path.join(im_outpath, "face", str(count+1)+".jpg"), im_face)
            cv2.imwrite(os.path.join(im_outpath, "left", str(count+1)+".jpg"), im_left)
            cv2.imwrite(os.path.join(im_outpath, "right", str(count+1)+".jpg"), im_right)
            
            save_name_face = os.path.join(person, "face", str(count+1) + ".jpg")
            save_name_left = os.path.join(person, "left", str(count+1) + ".jpg")
            save_name_right = os.path.join(person, "right", str(count+1) + ".jpg")
            save_gaze = ",".join(gaze.astype("str"))
            
            save_str = " ".join([save_name_face, save_name_left, save_name_right, save_gaze ])
            outfile.write(save_str + "\n")
        print("")
        outfile.close()

# def GetImage(image):
#         im = cv2.warpPerspective(image, W_mat, (int(imsize[0]), int(imsize[1])))
        # return im
    
def CropEye(im, imsize, lcorner, rcorner):
        try:
            im
        except:
            print("There is no image, please use GetImage first.")

        x, y = list(zip(lcorner, rcorner))
        
        center_x = np.mean(x)
        center_y = np.mean(y)

        width = np.abs(x[0] - x[1])*1.5
        times = width/60
        height = 36 * times
        
        print(im.shape)

        x1 = [max(center_x - width/2, 0), max(center_y - height/2, 0)]
        x2 = [min(x1[0] + width, imsize[0]), min(x1[1] + height, imsize[1])]
        im = im[int(x1[1]):int(x2[1]), int(x1[0]):int(x2[0])]
        print(type(im))
        print(im.shape)
        im = cv2.resize(im, (60, 36))
        return im
    
def CropFace(im, imsize, llcorner, rrcorner, lmouth, rmouth):
        try:
            im
        except:
            print("There is no image, please use GetImage first.")

        x, y = list(zip(llcorner, rrcorner, lmouth, rmouth))
        
        center_x = np.mean(x)
        center_y = np.mean(y)

        width = np.abs(x[0] - x[1])*1.5
        height = width

        x1 = [max(center_x - width/2, 0), max(center_y - height/2, 0)]
        x2 = [min(x1[0] + width, imsize[0]), min(x1[1] + height, imsize[1])]
        im = im[int(x1[1]):int(x2[1]), int(x1[0]):int(x2[0])]
        im = cv2.resize(im, (224, 224))
        return im, x1, x2
    

    
def AnnoDecode(anno_info):
    annotation = np.array(anno_info).astype("float32")
    out = {}
    out["gazepx"] = annotation [0:2]
    out["left_left_corner"] = annotation[2:4]
    out["left_right_corner"] = annotation[4:6]
    out["right_left_corner"] = annotation[6:8]
    out["right_right_corner"] = annotation[8:10]
    # out["headrotvectors"] = annotation[14:17]
    # out["headtransvectors"] = annotation[17:20]
    # out["facecenter"] = annotation[20:23]
    # out["target"] = annotation[23:26]
    return out


if __name__ == "__main__":
    ImageProcessing_MPII()
