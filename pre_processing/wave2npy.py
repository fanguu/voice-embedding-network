# -*- coding: utf-8 -*-
import os
import glob
import torch
import random
import shutil
import torchvision.utils as vutils
import webrtcvad
import matplotlib.pyplot as plt
import numpy as np
import csv
from mfcc import MFCC
from utils import rm_sil,get_fbank

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
 
    # 判断结果
    if not isExists:
        os.makedirs(path) 
 
        print(path+' 创建成功')
        return True
    else:
        print(path+' 目录已存在')
        return False

def get_dataset_files(data_dir, data_ext):
    data_list = []
    headers = ['filepath','name']
    vad_obj = webrtcvad.Vad(2)
    mfc_obj = MFCC(nfilt=64, lowerf=20., upperf=7200., samprate=16000, nfft=1024, wlen=0.025)
    # read data directory
    print("1")
    for root, dirs, filenames in os.walk(data_dir):      # 根目录, 子目录, 文件名
        for filename in filenames:
            if filename.endswith(data_ext):              # 校验文件后缀名
                filepath = os.path.join(root, filename)
                print(filepath)
                # so hacky, be careful! 
                folder = filepath[len(data_dir):].split('/')[1]
                print(folder)
                new_name = filename.split('.')[0]+".npy"
                print(new_name)
                new_path = '/media/fenger/DATA/1 datasets/data/'+folder+"/"
                new_filepath =  new_path+new_name
                mkdir(new_path)
                vad_voice = rm_sil(filepath, vad_obj)
                fbank = get_fbank(vad_voice, mfc_obj)
                #fbank = fbank.T[np.newaxis, ...]
                #fbank = torch.from_numpy(fbank.astype('float32'))
                np.save(new_filepath,fbank)
                print(fbank.shape)
                data_list.append({'filepath': new_filepath, 'name': folder})
                print('filepath', new_filepath,'name',folder)

    with open('test.csv','w') as f:
    	f_scv = csv.DictWriter(f,headers)
    	f_scv.writeheader()
    	f_scv.writerows(data_list)
        
def get_train_dataset(dirPath, destPath,rate):
    data_list = []
    headers = ['filepath','name']
    subDirs=os.listdir(dirPath)
    for my_dir in subDirs:
        tempDir=dirPath+'/'+my_dir+'/'
        if not os.path.exists(destPath+'/'+my_dir+'/'):
            os.mkdir(destPath+'/'+my_dir+'/')
        fs=os.listdir(tempDir)
        #print(len(fs))
        random.shuffle(fs)
        le=int(len(fs)*rate)  #这个可以修改划分比例
        for f in fs[:le]:
            shutil.copyfile(tempDir+f,destPath+'/'+my_dir+'/'+f)
            data_list.append({'filepath': destPath+'/'+my_dir+'/'+f, 'name': my_dir})
            print('filepath', destPath+'/'+my_dir+'/'+f, 'name', my_dir)
            
    with open('train.csv','w') as f:
    	f_scv = csv.DictWriter(f,headers)
    	f_scv.writeheader()
    	f_scv.writerows(data_list)

if __name__ == "__main__":
	#get_dataset_files('/media/fenger/DATA/1 datasets/cut','wav')
    get_train_dataset('/media/fenger/DATA/1 datasets/data','/media/fenger/DATA/1 datasets/train',0.8)
