import os
import torch
from HED_VGG16Net import VGG16NetHED
from loadDataset import HEDDatasetTrain,HEDDatasetTest
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
import argparse
from torchvision import transforms,utils
import cv2
import glob
import numpy as np

def get_file(path):
  files=[]
  for ext in ['jpg','png','jpeg','JPG']:
    files.extend(glob.glob(os.path.join(path,'*.{}'.format(ext))))
  return files

def MakeDir(path):
    if os.path.exists(path):
        return
    else:
        os.makedirs(path)
        
def short_side_resize(img,short_size):
  h,w = img.shape[0:2]
  if(w>h):
    new_h,new_w = short_size,int(short_size*w/h)
  else:
    new_h,new_w = int(short_size*h/w),short_size
  img = cv2.resize(img,(new_w,new_h))
  return img


def main(args):
    model = VGG16NetHED(pretrained=False)
    if args.checkpointPath:
        try:
            print("load model from {}".format(args.checkpointPath))
            Checkpoint = torch.load(args.checkpointPath)
            model.load_state_dict(Checkpoint['net'])
        except:
            print("checkpointPath error! Fail to Load the checkpoint!")
        
    else:
        print("checkpointPath error!  please cheack the  checkpoint Path...")
        
    GPU_list = [int(s) for s in args.gpu_list.split(',')]
    print('GPU_list:',GPU_list)
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    model = model.to(device)
    files = get_file(args.testImgPath)
    indx = 0
    with torch.no_grad():
        model.eval()
        for imgfile in files:
            indx+=1
            print("Processing the [{}/{}] image....".format(indx,len(files)))
            imagename = os.path.basename(imgfile)
            img = cv2.imdecode(np.fromfile(imgfile,dtype=np.uint8),-1)
            #img = short_side_resize(img,args.imageSize)
            h,w = img.shape[0:2]
            img = cv2.resize(img, (512, 512))
            img = img.reshape((1,img.shape[0],img.shape[1],3))
            img = torch.from_numpy(img)
            img = img.float().div(255)
            img = img.permute(0,3,1,2)
            img = img.to(device)
            output = model(img)
            outputImg = output[0,:,:,:]
            outputImg = outputImg.permute(1,2,0)
            outimage = outputImg.cpu().numpy()*255
            imagename = os.path.join(args.saveOutPath,imagename)
            outimage = cv2.resize(outimage,(w,h))
            cv2.imwrite(imagename,outimage)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "HED_TrainPerparams")
    parser.add_argument('--testImgPath',nargs='?',type=str,default='./demo',help='Path of the test image.')
    parser.add_argument('--saveOutPath',nargs='?',type=str,default='./demoout',help='Path to the save the test image result.')
    parser.add_argument('--checkpointPath',nargs='?',type=str,default='./savecheckpoint/58000_net.pkl',help='Path to the checkpoint.')
    parser.add_argument('--gpu_list',nargs='?',type=str,default='2')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_list
    MakeDir(args.saveOutPath)
    main(args)
