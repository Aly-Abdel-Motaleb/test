import os
import glob
import tqdm
import sys

import numpy as np
import skimage.io
import torch
import torch.utils.data

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from datasets.custom import custom
from datasets.kitti import KITTI
from engine.detector import Detector
from model.squeezedet import SqueezeDet
from utils.config import Config
from utils.model import load_model


def demo(cfg):
    # prepare configurations
    cfg.load_model = 'kaggle/working/SqueezeDet/models/model_55.pth'
    cfg.gpus = [-1]  # -1 to use CPU
    cfg.debug = 2  # to visualize detection boxes
    dataset = custom('val', cfg)
    cfg = Config().update_dataset_info(cfg, dataset)

    # preprocess image to match model's input resolution
    preprocess_func = dataset.preprocess
    del dataset

    # prepare model & detector
    model = SqueezeDet(cfg)
    model = load_model(model, cfg.load_model)
    detector = Detector(model.to(cfg.device), cfg)

    # prepare images
    sample_images_dir = 'kaggle/working/SqueezeDet/data/samples/kitti/training/image_2'
    sample_image_paths = glob.glob(os.path.join(sample_images_dir, '*.jpg'))

    # detection
    for path in tqdm.tqdm(sample_image_paths):
        image = skimage.io.imread(path).astype(np.float32)
        image_meta = {'image_id': os.path.basename(path)[:-4],
                      'orig_size': np.array(image.shape, dtype=np.int32)}

        image, image_meta, _ = preprocess_func(image, image_meta)
        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(cfg.device)
        image_meta = {k: torch.from_numpy(v).unsqueeze(0).to(cfg.device) if isinstance(v, np.ndarray)
                      else [v] for k, v in image_meta.items()}

        inp = {'image': image,
               'image_meta': image_meta}

        _ = detector.detect(inp)
