import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class VoiceEmbedNet(nn.Module):       # 'channels': [256, 384, 576, 864]
    def __init__(self, input_channel, channels, output_channel,num_class):
        super(VoiceEmbedNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=channels[0],
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(channels[0], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[0], channels[1], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[1], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[1], channels[2], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[2], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[2], channels[3], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[3], affine=True),
            
        )
        self.lstm = nn.LSTM(channels[3],hidden_size=4096,num_layers=1,batch_first=True)
        #self.fcn1_new = nn.Linear(channels[3],4096)
        self.fcn21 = nn.Linear(4096, num_class)
        

    def forward(self, x):
        x = self.model(x)

        x = torch.transpose(x, 1, 2) # 
        x,(h,v) =self.lstm(x,None)
        nn.ReLU(inplace=True)
        x = torch.transpose(x, 1, 2) #
        x = F.avg_pool1d(x, x.size()[2], stride=1)
        x = x.view(x.size()[0], -1)     # 平铺为一维

        x = self.fcn21(x)
        
        return F.log_softmax(x,dim=1)


class VoiceEmbedNet2(nn.Module):  # 'channels': [256, 384, 576, 864]
    def __init__(self, input_channel, channels, output_channel, num_class):
        super(VoiceEmbedNet2, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256, affine=True),
            nn.ReLU(inplace=True),
            # nn.functional.max_pool1d

            nn.Conv1d(256, 384, 3, 2, 1, bias=False),
            nn.BatchNorm1d(384, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 576, 3, 2, 1, bias=False),
            nn.BatchNorm1d(576, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(576, 864, 3, 2, 1, bias=False),
            nn.BatchNorm1d(864, affine=True),
            # nn.ReLU(inplace=True),
        )
        #self.lstm = nn.LSTM(256, hidden_size=4096, num_layers=1, batch_first=True)
        self.fcn_v21 = nn.Linear(864,4096)
        self.fcn_v22 = nn.Linear(4096, num_class)

    def forward(self, x):
        x = self.model(x)

        #x = torch.transpose(x, 1, 2)  #
        #x, _ = self.lstm(x, None)
        #nn.ReLU(inplace=True)
        #x = torch.transpose(x, 1, 2)  #
        x = F.avg_pool1d(x, x.size()[2], stride=1)
        x = x.view(x.size()[0], -1)  # 平铺为一维
        x = self.fcn_v21(x)
        x = self.fcn_v22(x)

        return F.log_softmax(x, dim=1)