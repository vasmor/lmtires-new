#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os
import sys
import traceback

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)

DEBUG = False  # Включить отладочный вывод: DEBUG = True


@hydra.main(config_path='../configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        if DEBUG:
            print(f"[DEBUG] indir: {predict_config.indir}")
            print(f"[DEBUG] outdir: {predict_config.outdir}")
            print(f"[DEBUG] model.path: {predict_config.model.path}")
            print(f"[DEBUG] model.checkpoint: {predict_config.model.checkpoint}")
            print(f"[DEBUG] dataset.img_suffix: {predict_config.dataset.img_suffix}")
            print(f"[DEBUG] device: {predict_config.get('device', 'cuda')}")
            print("[DEBUG] --- Начало инференса ---")
        if sys.platform != 'win32':
            if DEBUG:
                register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device("cpu")

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')

        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'

        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        if DEBUG:
            print(f"[DEBUG] Найдено изображений: {len(dataset)}")
        # Определяем режим online по типу датасета
        is_online = dataset.__class__.__name__ == 'InpaintingEvalOnlineDataset'
        if is_online:
            if DEBUG:
                print(f"[DEBUG] Примеры img_filenames: {dataset.img_filenames[:3]}")
        else:
            if DEBUG:
                print(f"[DEBUG] Примеры mask_filenames: {dataset.mask_filenames[:3]}")
                print(f"[DEBUG] Примеры img_filenames: {dataset.img_filenames[:3]}")
        for img_i in tqdm.trange(len(dataset)):
            if is_online:
                img_fname = dataset.img_filenames[img_i]
                img_base = os.path.splitext(img_fname[len(predict_config.indir):])[0]
                out_base = img_base + '_inpainted'
                cur_out_fname = os.path.join(
                    predict_config.outdir, 
                    out_base + out_ext
                )
            else:
                mask_fname = dataset.mask_filenames[img_i]
                mask_base = os.path.splitext(mask_fname[len(predict_config.indir):])[0]
                if mask_base.endswith('_mask'):
                    out_base = mask_base[:-5] + '_inpainted'
                else:
                    out_base = mask_base + '_inpainted'
                cur_out_fname = os.path.join(
                    predict_config.outdir, 
                    out_base + out_ext
                )
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            batch = default_collate([dataset[img_i]])
            if predict_config.get('refine', False):
                assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                # image unpadding is taken care of in the refiner, so that output image
                # is same size as the input image
                cur_res = refine_predict(batch, model, **predict_config.refiner)
                cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
            else:
                with torch.no_grad():
                    batch = move_to_device(batch, device)
                    batch['mask'] = (batch['mask'] > 0) * 1
                    batch = model(batch)                    
                    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                    unpad_to_size = batch.get('unpad_to_size', None)
                    if unpad_to_size is not None:
                        orig_height, orig_width = unpad_to_size
                        cur_res = cur_res[:orig_height, :orig_width]

            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname, cur_res)
        if DEBUG:
            print("[DEBUG] --- Инференс завершён ---")

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        if DEBUG:
            print(f'[ERROR] Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
