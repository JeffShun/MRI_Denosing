import os
import shutil
import SimpleITK as sitk
import numpy as np
import pathlib
import pydicom
import matplotlib.pylab as plt

def data_collate(src_dir, dst_dir):
    os.makedirs(dst_dir,exist_ok=True)
    for pat in os.listdir(src_dir):
        for seq in os.listdir(src_dir / pat):
            for scan in os.listdir(src_dir / pat / seq):
                os.makedirs(dst_dir / scan ,exist_ok=True)
                for data in os.listdir(src_dir / pat / seq / scan):
                    if (data.startswith(scan) or data.startswith("Temp")) and data.endswith(".dcm"):
                        shutil.move(str(src_dir / pat / seq / scan / data) , str(dst_dir / scan))

def correctify_dcm(data_dir):
    for dir in os.listdir(data_dir):
        dir_path = os.path.join(data_dir,dir)
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path,file)
            image = pydicom.dcmread(file_path)
            image.SpacingBetweenSlices = image.SliceThickness
            image.save_as(file_path)

def data_parser_and_grouping(data_dir, save_dir):
    info = dict()
    for scan in os.listdir(data_dir):
        try:
            data_path = str(data_dir / scan / pathlib.Path(scan+"0000.dcm"))
            image = pydicom.dcmread(data_path)
            CV_para = image[('0065'),('0021')].value
            if CV_para not in info:
                info[CV_para] = 1
            else:
                info[CV_para]+=1
            os.makedirs(save_dir / CV_para , exist_ok=True)
            shutil.move(str(data_dir / scan), str(save_dir / CV_para))
        except Exception as e:
            print(e)
            continue
    print(info)
        
def test_a_data(data_path):
    data = sitk.ReadImage(data_path)
    data_arr = np.array(data).reshape(data.GetSize()).squeeze()
    plt.imshow(data_arr,cmap="gray")
    plt.show()
    print(data_arr.shape)

if __name__ == "__main__":
    # src_dir = pathlib.Path(r"E:\ShenFile\02 项目数据\CV滤波数据\AI数据\AI数据\膝关节")
    # dst_dir = pathlib.Path(r"E:\ShenFile\02 项目数据\CV滤波数据\clean_data")
    # data_collate(src_dir, dst_dir)
    
    # src_dir = pathlib.Path(r"E:\ShenFile\02 项目数据\CV_Data\clean_data")
    # dst_dir = pathlib.Path(r"E:\ShenFile\02 项目数据\CV_Data\group_data")    
    # data_parser_and_grouping(src_dir, dst_dir)

    test_a_data(r"E:\ShenFile\02 项目数据\CV_Data\group_data\para2\63413\634130001.dcm")