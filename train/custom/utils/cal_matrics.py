"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pywt
import argparse
import pathlib
from argparse import ArgumentParser
from typing import Optional
import os
import numpy as np
from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pandas as pd

def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.array(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 2:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval
    ssim = structural_similarity(gt, pred, data_range=maxval)

    return ssim


METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        """
        Args:
            metric_funcs (dict): A dict where the keys are metric names and the
                values are Python functions for evaluating that metric.
        """
        self.metrics = {metric: Statistics() for metric in metric_funcs}
        self.metrics_data = {metric:[] for metric in metric_funcs}

    def push(self, pid, target, recons):
        for metric, func in METRIC_FUNCS.items():
            val = func(target, recons)
            self.metrics[metric].push(val)
            self.metrics_data[metric].append((pid, val))

    def means(self):
        return {metric: stat.mean() for metric, stat in self.metrics.items()}

    def stddevs(self):
        return {metric: stat.stddev() for metric, stat in self.metrics.items()}
    
    def save(self, save_dir):
        os.makedirs(save_dir,exist_ok=True)
        df = pd.DataFrame()
        # 遍历数据字典，将每种评价方式的数据添加为DataFrame的一列
        for method, values in self.metrics_data.items():
            labels, scores = zip(*values)
            df[method] = scores
        df['pid'] = labels
        df = df[['pid'] + [col for col in df.columns if col != 'pid']]
        csv_file_path = save_dir + "/metrics.csv"
        df.to_csv(csv_file_path, index=False)


    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return " ".join(
            f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}"
            for name in metric_names
        )

def evaluate(args):
    metrics = Metrics(METRIC_FUNCS)

    for sample in args.data_path.iterdir():
        pid = str(sample).split("\\")[-1]
        target = np.load(sample / "label.npy")
        recons = np.load(sample / "pred.npy")
        metrics.push(pid, target, recons)

    return metrics

def evaluate_HF(args):
    metrics = Metrics(METRIC_FUNCS)

    for sample in args.data_path.iterdir():
        target = np.load(sample / "label.npy")
        recons = np.load(sample / "pred.npy")

        # # DEBUG
        # import matplotlib.pylab as plt
        # plt.subplot(221)
        # plt.imshow(target)
        # plt.subplot(222)
        # plt.imshow(get_hfimg(target))
        # plt.subplot(223)
        # plt.imshow(recons)
        # plt.subplot(224)
        # plt.imshow(get_hfimg(recons))
        # plt.show()

        metrics.push(get_hfimg(target), get_hfimg(recons))

    return metrics

def get_hfimg(img):
    #设置高通滤波器
    rows, cols = img.shape
    crow,ccol = int(rows/2), int(cols/2)
    # 高通滤波得到高频成分
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift[crow-20:crow+20, ccol-20:ccol+20] = 0
    ishift = np.fft.ifftshift(fshift)
    hf = np.fft.ifft2(ishift)
    hf_img = np.abs(hf)
    return hf_img

def get_hfimg2(img):
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    return cD


if __name__ == "__main__":

    data_path = '../example/data/output/Paper-MW-ResUnet-DC4'
    
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--data_path",
        type=pathlib.Path,
        default=pathlib.Path(data_path+"/meta_datas"),
        help="Path to the ground truth data",
    )

    args = parser.parse_args()
    metrics = evaluate(args)
    metrics.save(data_path+"/metric_data")
    print(metrics)
