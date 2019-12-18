import os
import torch
import shutil
import numpy as np
import sys
from PIL import Image
from scipy.io import wavfile
from torch.utils.data.dataloader import default_collate
from vad import read_wave, write_wave, frame_generator, vad_collector

import logging
from logging import handlers


class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename, when='d', backCount=3):
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(logging.INFO)#设置日志级别
        console_handler = logging.StreamHandler(sys.stdout)#往屏幕上输出

        fmt = '%(asctime)s; %(levelname)s; %(message)s'
        format_str = logging.Formatter(fmt)#设置日志格式

        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')
        #往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：

        th.setFormatter(format_str)#设置文件里写入的格式
        console_handler.setFormatter(format_str)
        self.logger.addHandler(th)
        self.logger.addHandler(console_handler)

class Meter(object):
    # Computes and stores the average and current value
    def __init__(self, name, display, fmt=':f'):
        self.name = name
        self.display = display
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    # 返回对象的描述信息
    def __str__(self):
        fmtstr = '{name}:{' + self.display  + self.fmt + '},'
        return fmtstr.format(**self.__dict__)

def get_collate_fn(nframe_range):
    # collate_fn这个函数的输入就是一个batch size的数据, 音频数据为变长batch
    def collate_fn(batch):
        min_nframe, max_nframe = nframe_range
        assert min_nframe <= max_nframe, 'the wrong value in nframe'
        num_frame = np.random.randint(min_nframe, max_nframe+1)
        pt = np.random.randint(0, max_nframe-num_frame+1)
        batch = [(item[0][..., pt:pt+num_frame], item[1]) for item in batch]
        return default_collate(batch)
    return collate_fn

def cycle(dataloader):
    while True:
        for data, label in dataloader:
            yield data, label

def save_model(net, model_path):
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
       os.makedirs(model_dir)
    torch.save(net.state_dict(), model_path)

def rm_sil(voice_file, vad_obj):
    """
       This code snippet is basically taken from the repository
           'https://github.com/wiseman/py-webrtcvad'

       It removes the silence clips in a speech recording
    """
    audio, sample_rate = read_wave(voice_file)
    frames = frame_generator(20, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 20, 50, vad_obj, frames)

    if os.path.exists('tmp/'):
       shutil.rmtree('tmp/')
    os.makedirs('tmp/')

    wave_data = []
    for i, segment in enumerate(segments):
        segment_file = 'tmp/' + str(i) + '.wav'
        write_wave(segment_file, segment, sample_rate)
        wave_data.append(wavfile.read(segment_file)[1])
    shutil.rmtree('tmp/')

    if wave_data:
       vad_voice = np.concatenate(wave_data).astype('int16')
    return vad_voice

def get_fbank(voice, mfc_obj):
    # Extract log mel-spectrogra
    fbank = mfc_obj.sig2logspec(voice).astype('float32')

    # Mean and variance normalization of each mel-frequency 
    fbank = fbank - fbank.mean(axis=0)
    fbank = fbank / (fbank.std(axis=0)+np.finfo(np.float32).eps)

    # If the duration of a voice recording is less than 10 seconds (1000 frames),
    # repeat the recording until it is longer than 10 seconds and crop.
    full_frame_number = 1000
    init_frame_number = fbank.shape[0]
    while fbank.shape[0] < full_frame_number:
          fbank = np.append(fbank, fbank[0:init_frame_number], axis=0)
          fbank = fbank[0:full_frame_number,:]
    return fbank


