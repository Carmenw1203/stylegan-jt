import os
import cv2
import numpy as np
import pathlib
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create pictures of different resolutions')
    parser.add_argument('--input_dir', default='./data/FFHQ_data/FFHQ_128',type=str, help='save result pics')
    parser.add_argument('--output_dir', default='./data/FFHQ_data/FFHQ_',type=str, help='save result pics')
    # src_dir = './data/symbol_data/color_symbol_7k_128'
    args = parser.parse_args()
    src_dir = args.input_dir
    downsample_size = [8,16,32,64]
    target_dir = []
    for i in range(len(downsample_size)):
        target_dir.append(args.output_dir + str(downsample_size[i]))
        pathlib.Path(target_dir[i]).mkdir(parents=True,exist_ok=True)
        
    for item in os.listdir(src_dir):
        file_path = os.path.join(src_dir,item)
        img = cv2.imread(file_path)
        # print(type(img[0][0][1]))
        for i in range(len(downsample_size)):
            target_file_path = os.path.join(target_dir[i],item)
            target_img = cv2.resize(img,(downsample_size[i],downsample_size[i]),interpolation=cv2.INTER_AREA)
            cv2.imwrite(target_file_path,target_img)
