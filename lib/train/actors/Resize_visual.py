
import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np






import torch
import torchvision
import torchvision.transforms as T
from PIL import Image


def _read_image(self, image_file: str):
    if isinstance(image_file, str):
        im = cv.imread(image_file)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
    elif isinstance(image_file, list) and len(image_file) == 2:
        return decode_img(image_file[0], image_file[1])
    else:
        raise ValueError("type of image_file should be str or list")