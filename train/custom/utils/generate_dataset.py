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
    parser.add_argument('--data_path', type=str, default='./train_data/origin_data_para2')
    parser.add_argument('--save_path', type=str, default='./train_data/processed_data_para2')
    args = parser.parse_args()
    return args


def load_scans(dcm_path):
    reader = sitk.ImageSeriesReader()
    name = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(name)
    img = reader.Execute()
    return img


def gen_lst(save_path, task, all_pids):
    save_file = os.path.join(save_path, task+'.txt')
    data_list = glob.glob(os.path.join(save_path, '*.npz'))
    num = 0
    with open(save_file, 'w') as f:
        for pid in all_pids:
            data = os.path.join(save_path, pid+".npz")
            if data in data_list:
                num+=1
                f.writelines(data.replace("\\","/") + '\n')
    print('num of data: ', num)


def process_single(input):
    src_path, tgt_path, save_path, pid = input
    src_itk = sitk.ReadImage(src_path)
    tgt_itk = sitk.ReadImage(tgt_path)
    if src_itk.GetSize() == tgt_itk.GetSize():
        src = sitk.GetArrayFromImage(src_itk)[0]
        tgt = sitk.GetArrayFromImage(tgt_itk)[0]
        np.savez_compressed(os.path.join(save_path, f'{pid}.npz'), src=src, tgt=tgt)


if __name__ == '__main__':
    args = parse_args()
    for task in ["train", "valid"]:
        print("\nBegin gen %s data!"%(task))
        src_dir = os.path.join(args.data_path, task)
        save_path = args.save_path
        os.makedirs(save_path, exist_ok=True)
        src_imgs = []
        for scan in os.listdir(src_dir):
            for img in os.listdir(os.path.join(src_dir, scan)):
                if "Temp" not in img:
                    src_imgs.append(os.path.join(src_dir, scan, img).replace("\\","/"))
        inputs = []
        for tgt_path in tqdm(src_imgs):
            img_name = tgt_path.split("/")[-1]
            pid = img_name.replace(".dcm","")
            src_path = tgt_path.replace(img_name, "Temp_"+img_name)
            if not os.path.exists(src_path):
                print(src_path + " does not exist!")
                continue         
            inputs.append([src_path, tgt_path, save_path, pid])

        all_pids = [img.split("/")[-1].replace(".dcm","") for img in src_imgs]
        pool = Pool(8)
        pool.map(process_single, inputs)
        pool.close()
        pool.join()
        # 生成Dataset所需的数据列表
        gen_lst(save_path, task, all_pids)
