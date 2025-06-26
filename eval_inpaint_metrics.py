import os
import argparse
import csv
from glob import glob
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
from saicinpainting.evaluation.losses.ssim import SSIM
from saicinpainting.evaluation.losses.lpips import PerceptualLoss

def load_img(path, size=None):
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

def mse(img1, img2):
    return torch.mean((img1 - img2) ** 2).item()

def psnr(img1, img2):
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse_val))

def main():
    parser = argparse.ArgumentParser(description='Evaluate inpainting quality metrics')
    parser.add_argument('--originals_dir', type=str, required=True, help='Папка с оригиналами')
    parser.add_argument('--inpainted_dir', type=str, required=True, help='Папка с inpainted-результатами')
    parser.add_argument('--out_csv', type=str, default='metrics.csv', help='Файл для сохранения метрик')
    parser.add_argument('--device', type=str, default='cuda', help='cuda или cpu')
    args = parser.parse_args()

    originals = {os.path.splitext(os.path.basename(f))[0]: f for f in glob(os.path.join(args.originals_dir, '*'))}
    inpainted = {os.path.splitext(os.path.basename(f))[0]: f for f in glob(os.path.join(args.inpainted_dir, '*'))}
    common = sorted(set(originals) & set(inpainted))
    if not common:
        print('Нет совпадающих файлов!')
        return

    ssim_metric = SSIM().to(args.device)
    lpips_metric = PerceptualLoss(use_gpu=(args.device=='cuda')).to(args.device)

    results = []
    for name in tqdm(common, desc='Images'):
        img1 = load_img(originals[name]).to(args.device)
        img2 = load_img(inpainted[name], size=img1.shape[-2:]).to(args.device)
        with torch.no_grad():
            ssim_val = ssim_metric(img1, img2).item()
            lpips_val = lpips_metric(img1, img2, normalize=True).item()
        mse_val = mse(img1, img2)
        psnr_val = psnr(img1, img2)
        results.append({'name': name, 'psnr': psnr_val, 'ssim': ssim_val, 'mse': mse_val, 'lpips': lpips_val})

    # Сохраняем CSV
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'psnr', 'ssim', 'mse', 'lpips'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
        # Средние значения
        mean_row = {'name': 'MEAN'}
        for k in ['psnr', 'ssim', 'mse', 'lpips']:
            mean_row[k] = np.mean([r[k] for r in results])
        writer.writerow(mean_row)
    print(f'Готово! Сохранено в {args.out_csv}')

if __name__ == '__main__':
    main() 