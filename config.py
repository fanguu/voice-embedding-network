import string
from dataset import VoiceDataset
from utils import get_collate_fn

SYSTEM_PARAMTES = {
    'log_dir': './log/wav_class.log',
    'VE_model_dir': './pretrained_models/45000params.pth',
    
    
}

DATASET_PARAMETERS = {
    # meta data provided by voxceleb1 dataset
    'meta_file': 'data/vox1_meta.csv',   # face person

    # voice dataset
    'voice_dir': 'data/fbank',
    'voice_ext': 'npy',

    # train data includes the identities
    # whose names start with the characters of 'FGH...XYZ'
    'split': string.ascii_uppercase[5:],   # 生成大写字母

    # dataloader
    'voice_dataset': VoiceDataset,
    'batch_size': 128,
    'nframe_range': [300, 800],
    'workers_num': 8,
    'collate_fn': get_collate_fn,

    # test data
    'test_data': 'data/example_data/'
}


NETWORKS_PARAMETERS = {
    # VOICE EMBEDDING NETWORK (e)
    'lr': 0.0002,
    'beta1': 0.5,
    'beta2': 0.999,

    # MODE, use GPU or not
    'GPU': True,
}
