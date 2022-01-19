import numpy as np
import cv2
import os
import pathlib
import argparse

def train_demo(args):
    fps = 10
    size = (1960,1000)
    video = cv2.VideoWriter('./output/video_face/face_train_process.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
    file_list = os.listdir(args.src_dir)
    print(type(file_list))
    file_list.sort()
    print(type(file_list))
    # print(file_list)
    cnt = 0
    for item in file_list:
        cnt += 1
        file_path = os.path.join(args.src_dir,item)
        img = cv2.imread(file_path)
        # print(img.shape)
        # tmp2 = img[:,:,2]
        # tmp1 = img[:,:,1]
        # img[:,:,2] = tmp1
        # img[:,:,1] = tmp2
        # img[:,:,2] = 0
        # img[:,:,1] = 0
        # img[:,:,0] = 0
        # print(img.shape)
        img_resize = cv2.resize(img,size,interpolation=cv2.INTER_AREA)
        video.write(img_resize)
        if(cnt == 500):
            break
    video.release()
    cv2.destroyAllWindows()

def interpolation_demo(args):
    fps = 200
    size = (1960,1000)
    video = cv2.VideoWriter('./output/video_face/face_interpolation_process.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
    file_list = os.listdir(args.src_dir)
    file_list.sort()
    for i in range(len(file_list)):
        item = str(i)
        file_path = os.path.join(args.src_dir,'sample_'+item+'.png')
        img = cv2.imread(file_path)
        # tmp0 = img[:,:,0]
        # tmp2 = img[:,:,2]
        # img[:,:,0] = tmp2
        # img[:,:,2] = tmp0
        img_resize = cv2.resize(img,size,interpolation=cv2.INTER_AREA)
        video.write(img_resize)
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', default='./output/face_80w',type=str,help='checkpoint path')
    parser.add_argument('--type',default='trainprocess',type=str)#optional interpolations
    args = parser.parse_args()
    if(args.type == 'trainprocess'):
        train_demo(args)
    if(args.type == 'interpolation'):
        interpolation_demo(args)