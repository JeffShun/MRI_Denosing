import argparse
import glob
import os
import sys
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer.predictor import DenosingModel, DenosingPredictor

def parse_args():
    parser = argparse.ArgumentParser(description='Test MRI Denosing')

    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--input_path', default='../example/data/input/test', type=str)
    parser.add_argument('--output_path', default='../example/data/output/MW-ResUnet', type=str)

    parser.add_argument(
        '--model_file',
        type=str,
        # default='../train/checkpoints/trt_model/model.engine'
        default='../train/checkpoints/MW-ResUnet/60.pth'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='./test_config.yaml'
    )
    args = parser.parse_args()
    return args


def inference(predictor: DenosingPredictor, img: np.ndarray):
    pred_array = predictor.predict(img)
    return pred_array

def save_img(img, save_path):
    input_sos, label, pred_img = img
    ssim = structural_similarity(label, pred_img, data_range=label.max())
    input_sos = (input_sos-input_sos.min())/(input_sos.max()-input_sos.min())*255
    label = (label-label.min())/(label.max()-label.min())*255
    pred_img = (pred_img-pred_img.min())/(pred_img.max()-pred_img.min())*255
    diff_img = np.abs(pred_img - label)**2
    save_img = np.concatenate((input_sos, label, pred_img, diff_img),1)
    save_img = Image.fromarray(save_img.astype(np.uint8))
    draw = ImageDraw.Draw(save_img)
    font = ImageFont.truetype("arial.ttf", size=25)
    title = "SSIM: %.4f"%(ssim)
    draw.text((800, 285), title, fill=(255), font=font,stroke_fill='red')
    # save_path_part = save_path.split("\\")
    # save_path = save_path_part[0] + "/" + "{:.4f}".format(ssim) + "_" + save_path_part[1]
    save_img.save(save_path)


def main(input_path, output_path, device, args):
    # TODO: 适配参数输入
    model_denosing = DenosingModel(
        model_f=args.model_file,
        config_f=args.config_file,
    )
    predictor_denosing = DenosingPredictor(
        device=device,
        model=model_denosing,
    )

    os.makedirs(output_path, exist_ok=True)
    for f_name in tqdm(os.listdir(input_path)):
        # if f_name != "1000267_22.h5":
        #     continue
        f_path = os.path.join(input_path, f_name)
        with h5py.File(f_path, 'r') as f:
            noise_img = f['noise_img'][:]        
            clean_img = f['clean_img'][:]        

        pid = f_name.replace(".h5", "")
        pred_array = inference(predictor_denosing, noise_img)
        save_img([noise_img, clean_img, pred_array], os.path.join(output_path, f'{pid}.png'))

        meta_data_dir = os.path.join(output_path, "meta_datas", pid)
        os.makedirs(meta_data_dir, exist_ok=True)
        np.save(meta_data_dir + "/pred.npy", pred_array)
        np.save(meta_data_dir + "/label.npy", clean_img)

if __name__ == '__main__':
    args = parse_args()
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        device=args.device,
        args=args,
    )