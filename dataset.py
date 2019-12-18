import numpy as np
from PIL import Image
from torch.utils.data import Dataset
# 表示torch中Dataset的抽象类


def load_voice(voice_item):
    voice_data = np.load(voice_item['filepath'])
    voice_data = voice_data.T.astype('float32')
    #print(voice_data.shape)
    voice_label = voice_item['label_id']
    return voice_data, voice_label

class VoiceDataset(Dataset):
    def __init__(self, voice_list, nframe_range):
        self.voice_list = voice_list
        self.crop_nframe = nframe_range[1] #[300, 800]

    def __getitem__(self, index):
        voice_data, voice_label = load_voice(self.voice_list[index])
        a = voice_data
        #print(self.voice_list[index])
        #print(voice_data.shape)
        assert self.crop_nframe <= voice_data.shape[1]
        pt = np.random.randint(voice_data.shape[1] - self.crop_nframe + 1)
        voice_data = voice_data[:, pt:pt+self.crop_nframe]
        return voice_data, voice_label

    def __len__(self):
        return len(self.voice_list)
