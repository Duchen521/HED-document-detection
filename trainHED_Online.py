import os
import torch
from HED_VGG16Net import VGG16NetHED
from loadDataset import HEDDatasetTrain,HEDDatasetTest,HEDDatasetTrainOnLine
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
import argparse
from torchvision import transforms,utils
import cv2



def MakeDir(path):
    if os.path.exists(path):
        return
    else:
        os.makedirs(path)

def cross_entropy_loss(prediction, label):
    #print (label,label.max(),label.min())
    label = label.long()
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    #print ("num_positive, num_negative:",num_positive, num_negative)
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask, reduce=False)
    # return torch.sum(cost)
    return torch.mean(cost)


def trainHEDnet(train_loader,val_loader,model,optimizer,iterationsNum,device):
  imgs,gt_image = train_loader.getbatch(args.batch_size)
  imgs,gt_image = imgs.to(device),gt_image.to(device)
  optimizer.zero_grad()
  output = model(imgs)
  loss = cross_entropy_loss(output,gt_image)
  loss.backward()
  optimizer.step()
  if iterationsNum %50 == 0:
      print("Train iterations: [{}/{} ({:.0f}%)\t Loss:{:.6f}]".format(iterationsNum,
          args.max_iterations,100.*iterationsNum/args.max_iterations,loss.item()))
  if iterationsNum%2000 == 0:
      ValHEDnet(val_loader,model,iterationsNum,args.SaveOutImgPath,device)
  if iterationsNum %2000==0:
      state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'iterationsNum':iterationsNum}
      dir = os.path.join(args.SaveCheckpointPath,str(iterationsNum)+"_net.pkl")
      torch.save(state, dir)


def ValHEDnet(test_loader,model,iterationsNum,savepath,device,checkpoint=None):
    with torch.no_grad():
        model.eval()
        print(len(test_loader))
        for batch_idx,sample in enumerate(test_loader):
            imgs = sample['image']
            h,w = sample['height'],sample['width']
            imgs = imgs.to(device)
            output = model(imgs)
            outputImg = output[0,:,:,:]
            outputImg = outputImg.permute(1,2,0)
            outimage = outputImg.cpu().numpy()*255
            imagename = os.path.join(savepath,str(iterationsNum)+"_"+str(batch_idx)+".png")
            outimage = cv2.resize(outimage,(w,h))
            cv2.imwrite(imagename,outimage)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def main(args):
    model = VGG16NetHED(pretrained=True)
    start_iterationsNum = 0
    if args.RestartCheckpoint:
        Checkpoint = torch.load(args.RestartCheckpoint)
        model.load_state_dict(Checkpoint['net'])
        start_iterationsNum = Checkpoint['iterationsNum']
        
    GPU_list = [int(s) for s in args.gpu_list.split(',')]
    print('GPU_list:',GPU_list)
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    model = model.to(device)



    # model = torch.nn.DataParallel(model,device_ids=GPU_list)
    
    train_loader = HEDDatasetTrainOnLine(args.fgpath,args.bgpath,2)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)


    Test_data = HEDDatasetTest(data_rootdir =args.test_data_dir, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(Test_data,batch_size = 1,shuffle=False,num_workers=2)



    iterationsNum = start_iterationsNum
    while iterationsNum < args.max_iterations:
        
        trainHEDnet(train_loader,test_loader,model,optimizer,iterationsNum,device)
        iterationsNum += 1
        adjust_learning_rate(optimizer,iterationsNum)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "HED_TrainPerparams")
    parser.add_argument('--fgpath',nargs='?',type=str,default='./DATASET/source_image/foreground_images/',help='Path to forthground.')
    parser.add_argument('--bgpath',nargs='?',type=str,default='./DATASET/source_image/background_images/',help='Path to background.')
    parser.add_argument('--test_data_dir',nargs='?',type=str,default='./DATASET/dataset/',help='Path to the validation image set.')
    parser.add_argument('--SaveCheckpointPath',nargs='?',type=str,default='./saveModelTrainedOnline',help='Path to the save checkpoint.')
    parser.add_argument('--SaveOutImgPath',nargs='?',type=str,default='./saveImgTrainedOnline',help='Path to the save images.')
    parser.add_argument('--RestartCheckpoint',nargs='?',type=str,default='',help='Path to the restart checkpoint.')
    parser.add_argument('--max_iterations',nargs='?',type=int,default=60000,help='Num of train iterations.')
    parser.add_argument('--batch_size',nargs='?',type=int,default=6,help='Num of batch size.')
    parser.add_argument('--lr',nargs='?',type=float,default=0.001,help='Learning rate.')
    parser.add_argument('--gpu_list',nargs='?',type=str,default='3')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_list
    MakeDir(args.SaveCheckpointPath)
    MakeDir(args.SaveOutImgPath)
    main(args)
