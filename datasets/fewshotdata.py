
import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch



def load_attention(data_path):
    all_data = torch.load(os.path.join(data_path, 'train_data.pth'))
    all_label = torch.load(os.path.join(data_path, 'train_label.pth'))

    return all_data, all_label


class attention_FewShot(Dataset):
    def __init__(self, data_path, partition='train'):
        self.data, self.label = load_attention(data_path)
        self.partition = partition

    def __getitem__(self, item):
        data = self.data[item]
        label = self.label[item]
        return data, label

    def __len__(self):
        return self.data.shape[0]

class FewShotAttention(Dataset):
    def __init__(self, split, data_path, way, shot, fold):
        self.split = split
        self.data_path = data_path
        self.way = way
        self.shot = shot
        self.fold = fold
        self.true_path = os.path.join(self.data_path, f'{self.way}_{self.shot}_{self.fold}')
        fewshot_params = {
            'partition': 'train' if split in ['train', 'valid'] else 'test',
            "data_path": self.true_path
        }
        self.dataset = attention_FewShot(**fewshot_params)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        data, label = self.dataset.__getitem__(idx)
        return 'ModelNetFewShot', 'sample', (data, label)
        # return {'pc': pc, 'label': label.item()}

