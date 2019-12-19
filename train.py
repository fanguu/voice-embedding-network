# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn.functional as F
import numpy as np

from config import DATASET_PARAMETERS, NETWORKS_PARAMETERS, SYSTEM_PARAMTES
from torch.utils.data import DataLoader
from parse_dataset import get_dataset_voice

from network import VoiceEmbedNet2
from utils import Meter, cycle, save_model, get_collate_fn,Logger
from dataset import VoiceDataset

from tensorboardX import SummaryWriter
from torchsummary import summary


print('Parsing your dataset...')
voice_list, id_class_num = get_dataset_voice("./dataset/new_train.csv")
voice_dataset = VoiceDataset(voice_list,DATASET_PARAMETERS['nframe_range'])

print('Preparing the dataloaders...')
collate_fn = get_collate_fn(DATASET_PARAMETERS['nframe_range'])
voice_loader = DataLoader(voice_dataset, shuffle=True, drop_last=True,
                          batch_size=128,
                          num_workers=DATASET_PARAMETERS['workers_num'],  
                          collate_fn=collate_fn,
                          pin_memory=True)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
voice_iterator = iter(cycle(voice_loader))
model = VoiceEmbedNet2(64, [256, 384, 576, 864],64,1251)
# summary(model, (64,700))

#设置冻结层
for param in model.parameters():
    param.requires_grad = True
for param in model.model.parameters():
    param.requires_grad = False

opitmizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=0.0003,momentum=0.9)
#model.load_state_dict(torch.load(SYSTEM_PARAMTES['VE_model_dir']), strict=False)


log_dir ="./log/{0}_wav_class.log".format(time.strftime("%m_%d_%H_%M"))
log = Logger(log_dir).logger
writer = SummaryWriter()

print('Training models...')
if NETWORKS_PARAMETERS['GPU']:
    model.cuda()
model.train()

for it in range(50001):
    voice, voice_label = next(voice_iterator)
    if NETWORKS_PARAMETERS['GPU']: 
        voice, voice_label = voice.cuda(), voice_label.cuda()   
    predict = model(voice)
    loss = F.nll_loss(predict,voice_label)  # 输出层 用了log_softmax 则需要用这个误差函数
    pred = predict.argmax(dim=1)
    
    test_acc = (torch.eq(pred,voice_label).sum().float().item())/len(voice)
    writer.add_scalars('data/scalar_group', {"xsinx": loss,
                                             "xcosx": test_acc}, it)
    log_msg = 'batch_num:'+str(it)+" "+'loss:'+str(loss.item())+" "+'acc:'+str(test_acc)
    
    log.info(log_msg)

    if it % 10000 == 0 and it > 0:
        s_time = time.strftime("%Y-%m-%d,%H:%M")
        torch.save(model.state_dict(), s_time+'-'+str(it)+'-params.pth')
    
    opitmizer.zero_grad()
    loss.backward()
    opitmizer.step()

writer.export_scalars_to_json("./all_scalars.json")
writer.close()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
