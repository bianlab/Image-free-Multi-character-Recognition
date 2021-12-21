#coding:utf8
import numpy as np
import time
import cv2
import os
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse
import matplotlib.pyplot as plt
 
def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='D:/software/SPB_Data/crnn_w/lib/config/OWN_config.yaml')

    parser.add_argument('--checkpoint', type=str, default='D:/software/SPB_Data/crnn_w/output/OWN/crnn/checkpoint_846_acc_0.1424.pth ',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r',errors='ignore') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet

    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def recognition(config, img, model, converter, device):

    # github issues: https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec/issues/211
    h, w = img.shape
    # fisrt step: resize the height and width of image to (32, x)
    img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)

    # second step: keep the ratio of image's text same with training
    h, w = img.shape
    w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    # normalize
    img = img.astype(np.float32)
    img = img / 255. #- config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred

    #print('results: {0}'.format(sim_pred))

if __name__ == '__main__':

    config, args = parse_arg()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = crnn.get_crnn(config).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    started = time.time()

    image_path = "D:/software/SPB_Data/crnn_w/Synthetic_Chinese_License_Plates/pic/"
    #image_path = unicode(image_path1, "utf8")
    img_path_basic = os.listdir(image_path)
    txt_path = 'D:/software/SPB_Data/crnn_w/Synthetic_Chinese_License_Plates/test.txt'
    #txt_path = unicode(txt_path1, "utf8")

    print(torch.max(model.pattern_fc.weight),torch.min(model.pattern_fc.weight))

    count = 0
    ture_count = 0
    with open(txt_path,'r',errors='ignore') as f:
        filename = [file for num, file in enumerate(f.readlines())]
    file_path = []
    for char in filename:
        h,g=char.split('.')
        file_path.append(h)
    #print(file_path)

    for img_path in file_path:
        count += 1
        print(count,ture_count/count)
        gt = img_path
        img = cv2.imread(image_path+img_path+'.jpg')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

        result = recognition(config, img, model, converter, device)
        #print(result,gt)
        if(result==gt):
            ture_count += 1
    print(count , ture_count, (ture_count/count))

    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))



