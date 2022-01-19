import numpy as np
import cv2
import os
import pathlib
import argparse

def train_demo(args):
    fps = 50
    size = (1960,1000)
    video = cv2.VideoWriter('./output/video/train_process.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
    file_list = os.listdir(args.src_dir)
    print(type(file_list))
    file_list.sort()
    print(type(file_list))
    # print(file_list)
    for item in file_list:
        file_path = os.path.join(args.src_dir,item)
        img = cv2.imread(file_path)
        img_resize = cv2.resize(img,size,interpolation=cv2.INTER_AREA)
        video.write(img_resize)
    video.release()
    cv2.destroyAllWindows()

def interpolation_demo(args):
    fps = 200
    size = (1960,1000)
    video = cv2.VideoWriter('./output/video/interpolation_process.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
    file_list = os.listdir(args.src_dir)
    # print(type(file_list))
    file_list.sort()
    for i in range(len(file_list)):
        item = str(i)
        file_path = os.path.join(args.src_dir,'sample_'+item+'.png')
        img = cv2.imread(file_path)
        img_resize = cv2.resize(img,size,interpolation=cv2.INTER_AREA)
        video.write(img_resize)
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', default='./output/symbol',type=str,help='checkpoint path')
    parser.add_argument('--type',default='trainprocess',type=str)#optional interpolations
    args = parser.parse_args()
    if(args.type == 'trainprocess'):
        train_demo(args)
    if(args.type == 'interpolation'):
        interpolation_demo(args)