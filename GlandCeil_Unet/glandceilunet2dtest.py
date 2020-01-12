from __future__ import division
from unet2d.model_GlandCeil import unet2dModule
import numpy as np
import pandas as pd
import cv2
import os


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvmaskdata = pd.read_csv('./GlandsMask.csv')
    csvimagedata = pd.read_csv('./GlandsImage.csv')
    maskdata = csvmaskdata.iloc[:, :].values
    imagedata = csvimagedata.iloc[:, :].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    maskdata = maskdata[perm]

    unet2d = unet2dModule(512, 512, channels=3, costname="dice coefficient")
    unet2d.train(imagedata, maskdata, "./model/unet2dglandceil.pd",
                 "./log", 0.0005, 0.8, 1000, 2)


def predict(filePath, targetFileName):
    true_img = cv2.imread(filePath, cv2.IMREAD_COLOR)
    test_images = true_img.astype(np.float)
    # convert from [0:255] => [0.0:1.0]
    test_images = np.multiply(test_images, 1.0 / 255.0)
    unet2d = unet2dModule(512, 512, 3)
    predictvalue = unet2d.prediction("./model/unet2dglandceil.pd",
                                     test_images)
    cv2.imwrite("../mouth_picture/data/test/stardand/mask/mouth/"+targetFileName, predictvalue)

def getPredictValueByImg(img):
    img_w = img.shape[0]
    img_h = img.shape[1]
    resize_img = cv2.resize(img, (512, 512))
    test_images = resize_img.astype(np.float)
    # convert from [0:255] => [0.0:1.0]
    test_images = np.multiply(test_images, 1.0 / 255.0)
    unet2d = unet2dModule(512, 512, 3)
    predictvalue = unet2d.prediction("./model/unet2dglandceil.pd",
                                     test_images)
    return cv2.resize(predictvalue, (img_w, img_h))

def predictByFolder(folderPath):
    for root, dirs, files in os.walk(folderPath):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        # 遍历文件
        for f in files:
            if (f.startswith("original")):
                continue
            imgPath = os.path.join(root, f)
            print(imgPath)
            predict(imgPath, "mask_" + f.split('.')[0] + ".png")

def main(argv):
    if argv == 1:
        train()
    if argv == 2:
        #predict("C:/Users/liden/deep_learning/mouth_project/mouth_picture/data/test/stardand/image/mouth/mouth_gen_9_7811459.png", 'test_1.png')
        predictByFolder("../mouth_picture/data/test/stardand/image/mouth/")


if __name__ == "__main__":
    main(2)
