import argparse
import numpy as np
import imageio
import numpy as np
import cv2
import os
import yaml
import argparse

import matplotlib.pyplot as plt
from skimage import measure
from skimage import transform
from scipy.io import loadmat
from PIL import Image
import scipy.io as sio

def ReadTxt( path: str ) -> list:
    with open(path, "r", encoding='utf-8') as f:
        alls = []
        lines = f.readlines()
        for line in lines:
            data = list(map(str, line.strip().split(' ')))
            if len(data) > 1:
                alls.append( data )
        #print(alls)
    f.close()
    return alls


def main():
    pattern = loadmat("D:/software/MATLAB/R2016a/bin/crnn_w/1114_data.mat")
    # print(pattern["data"])
    txt_path = 'D:/software/MATLAB/R2016a/bin/crnn_w/experiment_1113/test_100_license.txt'
    img_path = 'D:/software/MATLAB/R2016a/bin/crnn_w/Synthetic_Chinese_License_Plates/pic/'
    alls = ReadTxt(txt_path)
    print( 'Number of images tested is:', len(alls) )
    #print(alls)
    measurement = np.zeros((100, 150))
    with open(txt_path, mode="r", encoding='utf-8') as f:
        for i in range(len(alls)):
            '''im1 = Image.open(img_path + str(alls[i][0])).convert('RGB')
            r,g,b = im1.split()
            bgr_img = Image.merge("RGB", [b, g, r])
            license_image1 = bgr_img.convert('L')
            #license_image1.show()
            license_image = (np.array(license_image1) / 255).astype('float32') # float32 灰度图像'''
            A = str(alls[i][0])
            im1 = cv2.imread(img_path + A,'r',encoding='utf-8')  # .convert('RGB')
            print(im1)
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            im1 = im1.astype('float32')
            license_image = im1/255

            print(license_image)
            for k in range(150):
              pattern_all = pattern["data"]
              C = pattern_all[k]
              temp = license_image.view(3072,1)*C    #license_image.flatten()
              measurement[i,k] = temp.sum()
        print(measurement)
        #np.savetxt("D:/software/MATLAB/R2016a/bin/crnn_w/1114.txt", measurement, fmt="%f", delimiter=",")
        sio.savemat('1114_new.mat',{'array':measurement})




if __name__ == '__main__':
    main()





