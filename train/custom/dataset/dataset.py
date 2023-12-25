"""data loader."""

import numpy as np
from torch.utils import data
from custom.utils.common_tools import *

class MyDataset(data.Dataset):
    def __init__(
            self,
            dst_list_file,
            transforms
    ):
        self.data_lst = self._load_files(dst_list_file)
        self._transforms = transforms

    def _load_files(self, file):
        data_list = []
        with open(file, 'r') as f:
            for line in f:
                data_list.append(line.strip())
        return data_list

    def __getitem__(self, idx):
        source_data = self._load_source_data(self.data_lst[idx])
        return source_data

    def __len__(self):
        return len(self.data_lst)

    def _load_source_data(self, file_name):
        data = np.load(file_name, allow_pickle=True)
        src_img = data['src']
        tgt_img = data['tgt']

        # transform前，数据必须转化为[C,H,W]的形状
        src_img = src_img[np.newaxis,:,:].astype(np.float32)
        tgt_img = tgt_img[np.newaxis,:,:].astype(np.float32)

        if self._transforms:
            src_img, tgt_img = self._transforms(src_img, tgt_img)
    
        # import matplotlib.pyplot as plt
        # src_arr = src_img.squeeze().numpy()
        # tgt_arr = tgt_img.squeeze().numpy()
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(src_arr)
        # plt.subplot(122)
        # plt.imshow(tgt_arr)
        # plt.show()

        return src_img, tgt_img


