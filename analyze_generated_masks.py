import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from saicinpainting.training.data.masks import get_mask_generator

# Путь к конфигу обучения
CONFIG_PATH = 'configs/training/your_config.yaml'
N = 1000  # Количество масок для анализа
IMG_SIZE = (3, 256, 256)  # Размер входного изображения (C, H, W)
OUT_DIR = 'outputs/mask_analysis'
os.makedirs(OUT_DIR, exist_ok=True)

def load_mask_gen_kwargs(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    mask_gen_kwargs = config.get('mask_gen_kwargs', {})
    return mask_gen_kwargs

def main():
    mask_gen_kwargs = load_mask_gen_kwargs(CONFIG_PATH)
    mask_gen = get_mask_generator('mixed', mask_gen_kwargs)
    areas = []
    for i in range(N):
        mask = mask_gen(np.zeros(IMG_SIZE, dtype=np.float32))
        area = mask.sum() / (IMG_SIZE[1] * IMG_SIZE[2])
        areas.append(area)
        if i < 10:
            plt.imsave(os.path.join(OUT_DIR, f'mask_{i}.png'), mask[0], cmap='gray')
    areas = np.array(areas)
    print(f'Средняя площадь маски: {areas.mean():.4f}')
    print(f'Минимальная площадь маски: {areas.min():.4f}')
    print(f'Максимальная площадь маски: {areas.max():.4f}')
    plt.figure()
    plt.hist(areas, bins=30)
    plt.title('Распределение площадей масок')
    plt.xlabel('Доля закрашенной области')
    plt.ylabel('Частота')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'mask_area_hist.png'))
    print(f'Гистограмма сохранена в {os.path.join(OUT_DIR, "mask_area_hist.png")}')

if __name__ == '__main__':
    main() 