import os
import jittor.transform as transform
from jittor.dataset.dataset import Dataset
import cv2

class MultiResolutionDataset(Dataset):
    def __init__(self, root_path, resolution=8):
        super().__init__()
        
        self.transform  = transform.Compose([
            transform.ToPILImage(),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.resolution = resolution
        self.train_img = []
        self.img_num = 0
        resolution_path = root_path+str(resolution)
        print(resolution_path)
        for item in os.listdir(resolution_path):
            img_path = os.path.join(resolution_path, item)
            if(not((img_path[-4:] == '.jpg') or (img_path[-4:] == '.png'))):
                continue
            img = cv2.imread(img_path)
            self.train_img.append(img)
            self.img_num += 1
        print(self.img_num)
        
    def __len__(self):
        return self.img_num
    
    def __getitem__(self, i):
        item = self.train_img[i]
        return self.transform(item)