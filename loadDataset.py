import os
import torch
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,utils
from online_generate_data import online_data_generate 
import threading
from queue import Queue
import time
import glob
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def get_file(path):
  files=[]
  for ext in ['jpg','png','jpeg','JPG']:
    files.extend(glob.glob(os.path.join(path,'*.{}'.format(ext))))
  return files



class HEDDatasetTrain(Dataset):
    def __init__(self, csv_file,rootdir, transform=None):
        """
        Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.label_frame = pd.read_csv(csv_file,encoding="utf_8")
        self.transform = transform
        self.rootdir = rootdir
    def __len__(self):
        return len(self.label_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.rootdir,self.label_frame.iloc[idx, 0][2:])
        edgeGT_name = os.path.join(self.rootdir,self.label_frame.iloc[idx, 1][2:])
        image=cv2.imdecode(np.fromfile(img_name,dtype=np.uint8),-1)
        GT_image=cv2.imdecode(np.fromfile(edgeGT_name,dtype=np.uint8),-1)
        image = cv2.resize(image, (512, 512))
        GT_image = cv2.resize(GT_image, (512, 512))
        _,GT_image = cv2.threshold(GT_image,1,1,cv2.THRESH_BINARY)
        GT_image = GT_image.reshape((GT_image.shape[0],GT_image.shape[1],1))
        GT_image = torch.from_numpy(GT_image)
        GT_image = GT_image.permute(2,0,1)
        cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
        image = image.reshape((image.shape[0],image.shape[1],3))
        sample = {'image': image, 'edge': GT_image}
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample

def multiThresh(fgpath,bgpath,fgImglist,bgImglist,fgImgNum,bgImgNum,everyThre,img_Q,GT_Q):
    for _ in range(everyThre):
        imgout,edgeOut = online_data_generate(fgpath,bgpath,fgImglist,bgImglist,fgImgNum,bgImgNum)
        img_Q.put(imgout)
        GT_Q.put(edgeOut)

class HEDDatasetTrainOnLine():
    def __init__(self,fgpath, bgpath,threadingNum = 1, transform=None):
        self.transform = transform
        self.fgpath = fgpath
        self.bgpath = bgpath
        self.fgImglist = os.listdir(fgpath)
        self.bgImglist = os.listdir(bgpath)
        self.fgImgNum = len(self.fgImglist)
        self.bgImgNum = len(self.bgImglist)
        self.threadingNum = threadingNum
        

    def getbatch(self,batchsize):
        GTbatch = []
        imgBtah = []
        img_Q = Queue()
        GT_Q = Queue()
        if batchsize%self.threadingNum!=0:
            print("batchsize should be divided with no remainder by threadingNum!!! ")
        everyThre = int(batchsize/self.threadingNum)
        threads = []
        for i in range(self.threadingNum):
          t = threading.Thread(target=multiThresh,args=(self.fgpath,self.bgpath,self.fgImglist,self.bgImglist,self.fgImgNum,self.bgImgNum,everyThre,img_Q,GT_Q,))
          t.start()
          #print("start:",i)
          threads.append(t)
          #imgout,edgeOut = online_data_generate(self.fgpath,self.bgpath,self.fgImglist,self.bgImglist,self.fgImgNum,self.bgImgNum)
          #GTbatch.append(imgout)
          #imgBtah.append(edgeOut)
        for thread in threads:
            thread.join()
        for _ in range(batchsize):
            GTbatch.append(GT_Q.get())
            imgBtah.append(img_Q.get())
        GTbatch = np.array(GTbatch)
        GTbatch = GTbatch.reshape((GTbatch.shape[0],GTbatch.shape[1],GTbatch.shape[2],1))
        GTbatch = torch.from_numpy(GTbatch)
        GTbatch = GTbatch.permute(0,3,1,2)
        GTbatch = GTbatch.float().div(255)
        
        imgBtah = np.array(imgBtah)
        imgBtah = imgBtah.reshape((imgBtah.shape[0],imgBtah.shape[1],imgBtah.shape[2],3))
        imgBtah = torch.from_numpy(imgBtah)
        imgBtah = imgBtah.permute(0,3,1,2)
        imgBtah = imgBtah.float().div(255)
        
        return imgBtah,GTbatch

class HEDDatasetTest(Dataset):
    def __init__(self,data_rootdir, transform=None):
        """
        Args:
        root_dir (string): Directory with all the test images.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.test_images = get_file(data_rootdir)
        self.transform = transform
    def __len__(self):
        return len(self.test_images)

    def __getitem__(self, idx):
        img_name = self.test_images[idx]
        # print(img_name)
        image=cv2.imdecode(np.fromfile(img_name,dtype=np.uint8),-1)
        h,w = image.shape[0:2]
        image = cv2.resize(image, (512, 512))
        cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
        image = image.reshape((image.shape[0],image.shape[1],3))
        sample = {'image': image,'height': h,'width': w}
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample

if __name__ =="__main__":
    fgpath = '/home1/dc/dcstuff/HED/CREATE_IMG/sourceImg/rect_images/'
    bgpath = '/home1/dc/dcstuff/HED/CREATE_IMG/sourceImg/background_images/'
    time_start=time.time()
    data = HEDDatasetTrainOnLine(fgpath,bgpath,2)
    imgBtah,GTbatch = data.getbatch(10)
    time_end=time.time()
    print('time cost',time_end-time_start,'s')
#    cv2.imwrite("11.jpg",GTbatch[0,:,:])
#    cv2.imwrite("00.jpg",imgBtah[0,:,:,:])
    print(GTbatch.shape,imgBtah.shape)
    print(GTbatch.dtype,imgBtah.dtype)
    # train_csv_file = './CREATE_IMG/HED_Dataset.csv'
    # rootdir = "./CREATE_IMG"
    # dataset = HEDDatasetTrain(csv_file=train_csv_file,rootdir =rootdir, transform=transforms.ToTensor())
    # print(len(dataset))
    # for i in range(len(dataset)):
    #     sample = dataset[i]
    #     if i==1:
    #         break
    #     print(i, sample["image"].size(),sample["edge"].shape)
    #     print(sample["image"])
    #     print(sample["edge"])


#    test_csv_file = './CREATE_IMG/HED_Test.csv'
#    rootdir = "./CREATE_IMG"
#    Testdataset = HEDDatasetTest(csv_file=test_csv_file,rootdir =rootdir, transform=transforms.ToTensor())
#    print(len(Testdataset))
#    for i in range(len(Testdataset)):
#        sample = Testdataset[i]
#        if i==1:
#            break
#        print(i, sample["image"].size())
#        print(sample["image"])

