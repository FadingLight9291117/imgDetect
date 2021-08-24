"""
处理数据集
"""
from dataset import Dataset
from utils import *
from pathlib import Path
import shutil

import numpy as np


dataset = Dataset.init_dir('./face')


imgL_paths = []
imgS_paths = []
imgE_paths = []
boxes = []
for data in dataset:
    imgL_path = data.imgL_path
    imgS_path = data.imgS_path
    box = data.box
    imgL_paths.append(imgL_path)
    imgS_paths.append(imgS_path)
    boxes.append(list(map(int, box)))

# 随机error img
for i in range(len(imgL_paths)):
    imgL_name = Path(imgL_paths[i]).name
    while True:
        idx = np.random.randint(len(imgL_path))
        rand_data = dataset[idx]
        if Path(rand_data.imgL_path).name != imgL_name:
            need_imgS = rand_data.imgS_path
            imgE_paths.append(need_imgS)
            break


save_dir = Path('./dataset')
imgL_dir = save_dir / 'imgL'
imgS_dir = save_dir / 'imgS'
imgE_dir = save_dir / 'imgE'
label = imgS_dir / 'label.json'

imgL_dir.mkdir(exist_ok=True)
imgS_dir.mkdir(exist_ok=True)
imgE_dir.mkdir(exist_ok=True)

for i in range(len(imgL_paths)):
    shutil.copy(imgL_paths[i], imgL_dir.joinpath(f'{i}.jpg').as_posix())
    shutil.copy(imgS_paths[i], imgS_dir.joinpath(f'{i}.jpg').as_posix())
    shutil.copy(imgE_paths[i], imgE_dir.joinpath(f'{i}.jpg').as_posix())

boxes = {
    'boxes': boxes
}

save2json(boxes, label)