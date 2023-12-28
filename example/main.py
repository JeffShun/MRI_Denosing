import argparse
import glob
import os
import sys
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity
import cv2
import matplotlib.font_manager as fm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer.predictor import DenosingModel, DenosingPredictor

def parse_args():
    parser = argparse.ArgumentParser(description='Test MRI Denosing')

    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--input_path', default='../example/data/input/para4', type=str)
    parser.add_argument('--output_path', default='../example/data/output/para4/temp', type=str)

    parser.add_argument(
        '--model_file',
        type=str,
        # default='../train/checkpoints/trt_model/model.engine'
        default='../train/checkpoints/Restormer/150.pth'
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


def save_img(imgs, save_path):
    noise_img, clean_img, pred_img = imgs
    ssim = structural_similarity(clean_img, pred_img, data_range=clean_img.max())
    noise_img = (noise_img-noise_img.min())/(noise_img.max()-noise_img.min())*255
    clean_img = (clean_img-clean_img.min())/(clean_img.max()-clean_img.min())*255
    pred_img = (pred_img-pred_img.min())/(pred_img.max()-pred_img.min())*255
    diff_img = np.abs(pred_img - clean_img).astype(np.uint8)*5
    noise_img = cv2.cvtColor(noise_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    clean_img = cv2.cvtColor(clean_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    pred_img = cv2.cvtColor(pred_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    diff_img = cv2.applyColorMap(diff_img, cv2.COLORMAP_BONE)[:,:,[2,1,0]]
    save_img = np.concatenate((noise_img, clean_img, pred_img, diff_img),1)
    save_img = Image.fromarray(save_img.astype(np.uint8))
    draw = ImageDraw.Draw(save_img)
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family="arial")), size=23)
    title = "SSIM: %.4f"%(ssim)
    w, h = save_img.size
    draw.text((int(w*0.75-160), int(h*0.89)), title, fill=(255,255,255), font=font,stroke_fill='white')
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

    for pid in tqdm(os.listdir(input_path)):
        # if f_name != "1000267_22":
        #     continue
        imgs_dir = os.path.join(input_path, pid)
        noise_imgs = [img for img in os.listdir(imgs_dir) if "Temp_" in img]
        for img in noise_imgs:
            noise_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(imgs_dir, img)))[0]
            clean_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(imgs_dir, img).replace("Temp_","")))[0]
            noise_img = (noise_img-noise_img.min())/(noise_img.max()-noise_img.min())
            clean_img = (clean_img-clean_img.min())/(clean_img.max()-clean_img.min())
            import time
            start = time.time()
            pred_array = inference(predictor_denosing, noise_img)
            print(time.time()-start)
            save_img([noise_img, clean_img, pred_array], os.path.join(output_path, f'{img.replace("Temp_","").replace(".dcm","")}.png'))
            meta_data_dir = os.path.join(output_path, "meta_datas", img.replace("Temp_",""))
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