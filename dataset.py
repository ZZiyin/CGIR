import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import data_augmentation
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from utils import *
from PIL import Image
from skimage.color import rgb2lab, rgb2gray

def load_img(img_path):
    out_np = np.asarray(Image.open(img_path))
    if (out_np.ndim == 2):
        out_np = np.tile(out_np[:, :, None], 3)  # channel*3
    return out_np
def normalize(data):
    return data/255.

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def prepare_data(data_path, patch_size, stride, aug_times=1):
    # train
    print('process training data')
    # scales = [1, 0.9, 0.8, 0.7]
    scales = [0.5,0.4]
    files = glob.glob(os.path.join(data_path, 'train2', '*jpg'))
    files.sort()
    h5f = h5py.File('train.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        # img = load_img(files[i])
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # img = np.expand_dims(img, 2)
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(w*scales[k]),int(h*scales[k])), interpolation=cv2.INTER_CUBIC)           
            Img = np.expand_dims(Img, 0)       
            Img = np.float32(normalize(Img))
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    train_num += 1
        if i==len(files)-1:
          plt.imshow(data.squeeze(0))
          plt.savefig('1.png')
    h5f.close()
    # label
    print('\nprocess label data')
    # scales = [1, 0.9, 0.8, 0.7]
    scales = [0.5, 0.4]
    files = glob.glob(os.path.join(data_path, 'train2', '*.jpg'))
    files.sort()
    h5f = h5py.File('label.h5', 'w')
    label_num = 0
    for i in range(len(files)):
        # img = cv2.imread(files[i])
        img = load_img(files[i])
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(w * scales[k]), int(h * scales[k])), interpolation=cv2.INTER_CUBIC)
            # Img = np.expand_dims(Img[:, :, :].copy(), 0)
            Img = Img.transpose(2,0,1)       
            Img = np.float32(normalize(Img))          
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3] * aug_times))
            for n in range(patches.shape[3]):
                data = patches[:, :, :, n].copy()
                h5f.create_dataset(str(label_num), data=data)
                label_num += 1
                for m in range(aug_times - 1):
                    data_aug = data_augmentation(data, np.random.randint(1, 8))
                    h5f.create_dataset(str(label_num) + "_aug_%d" % (m + 1), data=data_aug)
                    label_num += 1
        if i==len(files)-1:
          plt.imshow(data.transpose(1,2,0))
          plt.savefig('2.png')
    h5f.close()
    # # val
    # print('\nprocess validation data')
    # files.clear()
    # files = glob.glob(os.path.join(data_path, 'Set12', '*.png'))
    # files.sort()
    # h5f = h5py.File('val.h5', 'w')
    # val_num = 0
    # for i in range(len(files)):
    #     print("file: %s" % files[i])
    #     img = cv2.imread(files[i])
    #     img = np.expand_dims(img[:,:,0], 0)
    #     img = np.float32(normalize(img))
    #     h5f.create_dataset(str(val_num), data=img)
    #     val_num += 1
    # h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('label set, # samples %d\n' % label_num)
    # print('val set, # samples %d\n' % val_num)



class Dataset_new(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset_new, self).__init__()
        self.train = train
        if self.train:
            h5f = h5py.File('train.h5', 'r')
            h5f2 = h5py.File('label.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
            h5f2 = h5py.File('label.h5', 'r')
        self.keys = list(h5f.keys())
        self.keys_label = list(h5f2.keys())
        # random.shuffle(self.keys)
        # random.shuffle(self.keys_label)
        h5f.close()
        h5f2.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File('train.h5', 'r')
            h5f2 = h5py.File('label.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
            h5f2 = h5py.File('label.h5', 'r')
        key = self.keys[index]
        key_label = self.keys_label[index]
        data = np.array(h5f[key])
        label = np.array(h5f2[key_label])
        
          
        
        # img_lab = rgb2lab(label.transpose(1,2,0))
        # img_ab = img_lab[:, :, 1:3]
        # img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))
        # img_original = rgb2lab(label.transpose(1,2,0))[:,:,0]-50.
        # img_original = torch.from_numpy(img_original)
        h5f.close()
        h5f2.close()
        return data, label 
  
      #  return torch.Tensor(data), torch.Tensor(label)

class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)
