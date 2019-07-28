import torch
import torch.nn as nn
import torchvision
import numpy as np


# print(VGG16)
# model = VGG16.features
# print(model)
class VGG16NetHED(nn.Module):

    def __init__(self,pretrained=False):
        super(VGG16NetHED,self).__init__()
        self.VGG16 = torchvision.models.vgg16(pretrained=pretrained).features
        self.conv1 = self.VGG16[0:4]
        self.score1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)##512

        self.conv2 = self.VGG16[4:9]
        self.score2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)##256

        self.conv3 = self.VGG16[9:16]
        self.score3 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)##128

        self.conv4 = self.VGG16[16:23]
        self.score4 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)##64

        self.conv5 = self.VGG16[23:30]
        self.score5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)##32

        self.moduleCombine = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
			torch.nn.Sigmoid()
		)

    def forward(self,x):
        tensorVggOne = self.conv1(x)
        tensorVggTwo = self.conv2(tensorVggOne)
        tensorVggThr = self.conv3(tensorVggTwo)
        tensorVggFou = self.conv4(tensorVggThr)
        tensorVggFiv = self.conv5(tensorVggFou)

        tensorScoreOne = self.score1(tensorVggOne)
        tensorScoreTwo = self.score2(tensorVggTwo)
        tensorScoreThr = self.score3(tensorVggThr)
        tensorScoreFou = self.score4(tensorVggFou)
        tensorScoreFiv = self.score5(tensorVggFiv)

        tensorScoreOne = torch.nn.functional.interpolate(input=tensorScoreOne, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        tensorScoreTwo = torch.nn.functional.interpolate(input=tensorScoreTwo, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        tensorScoreThr = torch.nn.functional.interpolate(input=tensorScoreThr, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        tensorScoreFou = torch.nn.functional.interpolate(input=tensorScoreFou, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        tensorScoreFiv = torch.nn.functional.interpolate(input=tensorScoreFiv, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return self.moduleCombine(torch.cat([ tensorScoreOne, tensorScoreTwo, tensorScoreThr, tensorScoreFou, tensorScoreFiv ],1))

if __name__ =='__main__':
    HED = VGG16NetHED()
    print(HED)
    x = torch.randn(1,3,512,512)
    out = HED(x)
    print(out.shape)
