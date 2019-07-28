# -*- coding: UTF-8 -*-
import os
import pandas as pd
import argparse


def getTrainList(args):
    savePath = args.save_csv
    datasetPath = args.dataset_path
    allfilelist = []
    imgPath = os.path.join(datasetPath,'image')
    files = os.listdir(imgPath)
    for file in files:
        imageFile = os.path.join(imgPath,file)

        edgeGTFile = os.path.join(datasetPath+"edgeGT",file.split('.')[0]+".bmp")

        if not os.path.exists(edgeGTFile):
            continue
        imageFilesave = os.path.join("image",file)
        edgeGTFilesave = os.path.join("edgeGT",file.split('.')[0]+".bmp")
        filelist = [imageFilesave,edgeGTFilesave]
        allfilelist.append(filelist)
        print(file)
    datafile = pd.DataFrame(data = allfilelist)
    datafile.to_csv(savePath,index=False,header=False,encoding="utf_8_Sig")


#def getTestList(datasetPath,savePath):
#    allfilelist = []
#    imgPath = os.path.join(datasetPath)
#    files = os.listdir(imgPath)
#    for file in files:
#        imageFile = os.path.join(imgPath,file)
#        filelist = [imageFile]
#        allfilelist.append(filelist)
#        print(file)
#    datafile = pd.DataFrame(data = allfilelist)
#    datafile.to_csv(savePath,index=False,header=False,encoding="utf_8_Sig")

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description = "HED_Data_Generate")
    parser.add_argument('--dataset_path',nargs='?',type=str,default='../DATASET/dataset/',help='Path to dataset.')
    parser.add_argument('--save_csv',nargs='?',type=str,default='../DATASET/dataset/HED_Dataset.csv',help='saved csvfile name.')
    args = parser.parse_args()
    getTrainList(args)

#    testSavePath = "HED_Test.csv"
#    testDatasetPath = "./testData/"
#
#    getTestList(testDatasetPath,testSavePath)
