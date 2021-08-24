import sys

sys.path.append('..')
import json
from pathlib import Path
from dataclasses import dataclass
import logging

import numpy as np
import cv2

__all__ = ['Dataset']

from utils import Timer

logger = logging.getLogger(__name__)


@dataclass
class _Data:
    imgL_path: str
    imgS_path: str
    box: np.ndarray
    imgL: np.ndarray = None
    imgS: np.ndarray = None

    def _trans_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img / 255
        return img

    def _getImg(self, img_path, trans=True):
        img = cv2.imread(img_path)
        if trans:
            img = self._trans_img(img)
        return img

    def getImgL(self, trans=True):
        if self.imgL is not None:
            return self.imgL
        self.imgL = self._getImg(self.imgL_path, trans)
        return self.imgL

    def getImgS(self, trans=True):
        if self.imgS is not None:
            return self.imgS
        self.imgS = self._getImg(self.imgS_path, trans)
        return self.imgS

    def img_compose_cv2(self, img_path, quality, tmp_path):
        """
            opencv
        """

        img = cv2.imread(img_path)
        # tmp_path = Path('./tmp')
        # tmp_path.mkdir(exist_ok=True)
        # save_path = tmp_path / f'{tmpName}_{quality}.jpg'
        cv2.imwrite(tmp_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

        # img_res = cv2.imread(str(save_path))

    def load(self, trans=True, compress=True):
        quality = np.random.randint(2, 5) * 10
        if compress and quality != 100:
            tmp = cv2.imread(self.imgL_path)

            tmp_path = Path('./tmp') / f'{Path(self.imgS_path).stem}_{quality}.jpg'
            tmp_path.parent.mkdir(exist_ok=True)

            self.img_compose_cv2(self.imgS_path, quality, tmp_path.as_posix())
            self.imgS_path = tmp_path.as_posix()

        self.imgL = self._getImg(self.imgL_path, trans)
        self.imgS = self._getImg(self.imgS_path, trans)

        return self

    def clear(self):
        self.imgL = None
        self.imgS = None


class Dataset:
    @classmethod
    def init_dir(cls, data_dir, load=False):
        data_dir = Path(data_dir)
        imgL_dir = data_dir / 'imgL'
        imgS_dir = data_dir / 'imgS'
        label_path = data_dir / 'label.json'
        return cls(imgL_dir.__str__(), imgS_dir.__str__(), label_path, load)

    def __init__(self, imgL_dir, imgS_dir, label_path, load=False):
        if not Path(imgL_dir).exists() or not Path(imgS_dir).exists():
            logger.error('路径不存在')
            raise Exception
        self.load = load
        self.imgL_dir = imgL_dir
        self.imgS_dir = imgS_dir
        with open(label_path, encoding='utf-8') as f:
            self.labels = json.load(f)
        logger.debug(
            f'==> dataset loading finish @{str(Path(label_path).absolute())}, n={len(self.labels)}')

    def randOne(self):
        return self.randN(1, t=True)

    @Timer.timer
    def randN(self, num, t=True):
        if num > len(self):
            num = len(self)
            logger.warning(f'num: {num} 超出数据集长度.')
        if t is True:
            rand_idxs = np.random.choice(a=len(self), size=num, replace=False)
            datas = [self[i] for i in rand_idxs]
        else:
            datas = self[:num]
        datas = [data.load() for data in datas]
        return datas

    def _label2Data(self, label):
        imgL_name = label['imgL']
        imgS_name = label['imgS']
        imgL_path = Path(self.imgL_dir) / imgL_name
        imgS_path = Path(self.imgS_dir) / imgS_name
        box = label['box']
        box = np.array(box)
        return _Data(imgL_path.__str__(), imgS_path.__str__(), box)

    def __getitem__(self, i):
        label = self.labels[i]
        if type(label) is list:
            return [self._label2Data(l) for l in label]
        else:
            return self._label2Data(label)

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    import sys

    sys.path.append('../')  # !!!!!关于导包路径的问题终于搞懂了

    import matplotlib.pyplot as plt
    from PIL import Image
    from PIL.ImageDraw import ImageDraw

    imgL_path = '../data/widerface/imgL'
    imgS_path = '../data/widerface/imgS'
    label_path = '../data/widerface/label.json'

    dataset = Dataset(imgL_path, imgS_path, label_path)

    data = dataset.randOne()
    print(data.box)
    p1 = plt.subplot(311)
    p2 = plt.subplot(312)
    p3 = plt.subplot(313)

    p2.imshow(data.imgS)

    im = Image.fromarray(data.imgL)
    crop = im.crop(data.box)
    p3.imshow(crop)

    imgL = Image.fromarray(data.imgL)
    imgLD = ImageDraw(imgL)
    imgLD.rectangle(data.box.tolist())
    p1.imshow(imgL)

    plt.show()
