import os
import cv2
import numpy as np
import pathlib

if __name__ == '__main__':
    # src_dir = './data/symbol_data/color_symbol_7k_128'
    src_dir = './data/FFHQ_data/FFHQ_128'
    downsample_size = [8,16,32,64]
    target_dir = []
    for i in range(len(downsample_size)):
        target_dir.append('./data/FFHQ_data/FFHQ_' + str(downsample_size[i]))
        pathlib.Path(target_dir[i]).mkdir(parents=True,exist_ok=True)
        
    for item in os.listdir(src_dir):
        file_path = os.path.join(src_dir,item)
        img = cv2.imread(file_path)
        # print(type(img[0][0][1]))
        for i in range(len(downsample_size)):
            target_file_path = os.path.join(target_dir[i],item)
            target_img = cv2.resize(img,(downsample_size[i],downsample_size[i]),interpolation=cv2.INTER_AREA)
            cv2.imwrite(target_file_path,target_img)
