import numpy as np
import os

import cv2 as cv
from PIL import Image

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.jpg']

def load_image(file):
    
  
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None,data_aug=None,ms=None):
        self.images_root = os.path.join(root, 'images')
        
        self.labels_root = os.path.join(root, 'labels')
        
        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
     
        self.filenames.sort()
       
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.scale=[1/2,3/4,1,5/4,3/2]
        self.ms=ms

    def __getitem__(self, index):
       
        filename = self.filenames[index]
        degree = int(np.random.uniform(0,360,1))
        ind=int(np.random.uniform(0,5,1))
        
        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            
            image = load_image(f).convert('RGB')                         
            if self.ms==1:
                s1,s2,s3=np.shape(image)
                image=image.resize((int(s1*self.scale[ind]),int(s2*self.scale[ind])))                
            if degree>180:
                image=image.transpose(Image.FLIP_LEFT_RIGHT)
                image=image.rotate(10)
           
           
        with open(image_path(self.labels_root, filename, '.jpg'), 'rb') as f:
            
            label = load_image(f).convert('P')                      
            if self.ms==1:
                label=label.resize((int(s1*self.scale[ind]),int(s2*self.scale[ind])))   
            if degree>180:
               label=label.transpose(Image.FLIP_LEFT_RIGHT)
               label=label.rotate(10)

        if self.input_transform is not None:
            image = self.input_transform(image)
     
        if self.target_transform is not None:
            label = self.target_transform(label)         
       
        save_path=(self.images_root+filename)
        return image, label,save_path

    def __len__(self):
        
        return len(self.filenames)
        

class test_set(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        
        
        self.filenames = [image_basename(f)
            for f in os.listdir(self.images_root) if is_image(f)]
     
        self.filenames.sort()
        
        self.input_transform = input_transform
   
        
    def __getitem__(self, index):
       
        filename = self.filenames[index]
     
        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')

        if self.input_transform is not None:
            image = self.input_transform(image)
     
        
        save_path=(self.images_root+filename)
        return image, save_path

    def __len__(self):
        
        return len(self.filenames)


