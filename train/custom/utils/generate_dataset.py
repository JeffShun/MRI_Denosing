"""生成模型输入数据."""

import argparse
import glob
import os

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from multiprocessing import Pool

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./train_data/origin_data')
    parser.add_argument('--save_path', type=str, default='./train_data/processed_data')
    args = parser.parse_args()
    return args


def load_scans(dcm_path):
    reader = sitk.ImageSeriesReader()
    name = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(name)
    img = reader.Execute()
    return img


def gen_lst(tgt_path, task, processed_pids):
    save_file = os.path.join(tgt_path, task+'.lst')
    data_list = glob.glob(os.path.join(tgt_path, '*.npz'))
    num = 0
    with open(save_file, 'w') as f:
        for pid in processed_pids:
            data = os.path.join(tgt_path, pid+".npz")
            if data in data_list:
                num+=1
                f.writelines(data + '\r\n')
    print('num of data: ', num)


def process_single(input):
    src_path, tgt_path, save_path, pid = input
    src_itk = sitk.ReadImage(src_path)
    tgt_itk = sitk.ReadImage(tgt_path)
    if src_itk.GetSize() == tgt_itk.GetSize():
        src = sitk.GetArrayFromImage(src_itk)
        tgt = sitk.GetArrayFromImage(tgt_itk)
        np.savez_compressed(os.path.join(save_path, f'{pid}.npz'), src=src, tgt=tgt)



if __name__ == '__main__':
    args = parse_args()
    src_path = args.src_path
    for task in ["train", "valid"]:
        print("\nBegin gen %s data!"%(task))
        src_dir = os.path.join(args.data_path, task, "src_nii")
        tgt_dir = os.path.join(args.data_path, task, "tgt_nii")
        save_path = args.save_path
        os.makedirs(tgt_path, exist_ok=True)
        inputs = []
        for pid in tqdm(os.listdir(src_dir)):
            src_path = os.path.join(src_dir, pid)
            tgt_path = os.path.join(tgt_dir, pid)           
            pid = pid.replace('.nii.gz', '') 
            inputs.append([src_path, tgt_path, save_path, pid])
        processed_pids = [pid.replace('.nii.gz', '') for pid in os.listdir(tgt_dir)]
        pool = Pool(8)
        pool.map(process_single, inputs)
        pool.close()
        pool.join()
        # 生成Dataset所需的数据列表
        gen_lst(save_path, task, processed_pids)
