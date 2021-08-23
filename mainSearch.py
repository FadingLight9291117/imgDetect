from pathlib import Path
import json
from typing import List, Dict

import cv2
from easydict import EasyDict as edict
import numpy as np
from PIL import Image, ImageDraw

from exceptions import RoiException, ImgException
from utils import json2dict, Timer

config = json2dict('config.json')

sub_method = eval(config['sub_method'])
enable_adapt = config['enable_adapt']
K = config['K']
mae_thresh = config['mae_thresh']
mae_mean = config['mae_mean']
mae_std = config['mae_std']


def main_search(imgL: np.ndarray,
                imgS: np.ndarray,
                roi,
                sub_method=sub_method,
                enable_adapt=enable_adapt,
                K=K,
                mae_thresh=mae_thresh,
                mae_mean=mae_mean,
                mae_std=mae_std) -> Dict:
    """
    1. 异常处理
    1.1 imgL or img = None
    1.2 roi = None or []
    1.3 roi 元素个数不是4
    1.4 roi 左上角左边大于右下角坐标
    1.5 roi 两坐标在同一直线
    1.6 roi 两坐标重合
    1.7 roi 两坐标都出同一边的界
    1.8 imgL比imgS小
    1.9 roi比imgS小

    2.roi规整
    2.1 roi越界；

    3. img转换
    3.1 是否灰度
    3.2 是否float
    3.3 是否 / 255

    4. match template

    5. 计算box

    6. 计算置信度

    7. return result

    params:
        imgL: np.ndarray 大图矩阵
        imgS: np.ndarray 小图矩阵
        roi:  List[4] 大图搜索范围
        method:

    return:
        dict{
            box: List,
            conf: float,
        }
    """

    # 1. 异常处理
    if imgL is None or imgS is None:
        raise ImgException('大图或者小图为None')
    if imgL.size < imgS.size:
        raise ImgException('大图比小图小')

    x1, y1, x2, y2 = roi
    h_l, w_l = imgL.shape[:2]
    h_s, w_s = imgS.shape[:2]

    if roi is None:
        raise RoiException('roi为None')
    if len(roi) != 4:
        raise RoiException('roi元素个数不为4')
    if x1 >= x2 or y1 >= y2:
        raise RoiException('roi两坐标不合法，或左上角坐标大于右下角坐标，或roi面积为0')
    if x1 < 0 or y1 < 0 or x2 > w_l or y2 > h_l:
        raise RoiException('roi超出大图范围')
    if x2 - x1 < w_s or y2 - y1 < h_s:
        raise RoiException('roi比小图小')

    # 2. roi规整
    x1 = bound(x1, 0, w_l)
    y1 = bound(y1, 0, h_l)
    x2 = bound(x2, 0, w_l)
    y2 = bound(y2, 0, h_l)

    # 3. img转换
    imgL_n = imgL[y1:y2, x1:x2]

    rate = adaptive_rate(imgS, enable_adapt, K)
    imgL_n = _trans_img(imgL_n, rate)
    imgS_n = _trans_img(imgS, rate)

    # 4. match template
    coeff = cv2.matchTemplate(imgL_n, imgS_n, sub_method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(coeff)
    if sub_method == cv2.TM_SQDIFF or sub_method == cv2.TM_SQDIFF_NORMED:
        box_x1, box_y1 = min_loc
    else:
        box_x1, box_y1 = max_loc

    # 6. 计算置信度
    conf = compute_conf(imgL_n[box_y1:box_y1 + imgS_n.shape[0], box_x1:box_x1 + imgS_n.shape[1]],
                        imgS_n, mae_mean, mae_std)

    # 5. 计算box
    box_x1 = _inv_trans(box_x1, rate)
    box_y1 = _inv_trans(box_y1, rate)
    box_x1 += x1
    box_y1 += y1
    box_x2 = box_x1 + w_s
    box_y2 = box_y1 + h_s
    box_x1 = bound(box_x1, 0, w_l)
    box_x2 = bound(box_x2, 0, w_l)
    box_y1 = bound(box_y1, 0, h_l)
    box_y2 = bound(box_y2, 0, h_l)

    box = [box_x1, box_y1, box_x2, box_y2]

    # 7. return result
    res = {
        'box': box,
        'conf': conf
    }

    return res


def _inv_trans(a, rate):
    a = int(a * rate)
    return a


def adaptive_rate(imgS, b_rate, K=50):
    rate = 1
    if b_rate == True:
        h, w = imgS.shape[:2]
        rate = np.min([h / K, w / K])
        rate = np.max([rate, 1])
        # rate = int(rate + 0.5)
        # rate = bound(rate, 3, 4)
        # a = int(np.log2(rate))
        # rate = np.power(2, a + 1)
    return rate


def _trans_img(img, rate):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)
    img /= 255

    if rate != 1:
        h, w = img.shape[:2]
        img = cv2.resize(
            img, (int(w // rate), int(h // rate)), cv2.INTER_LINEAR)
        # show_img(img)
    return img


def compute_conf(img1, img2, mae_mean, mae_std):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if h1 != h2 or w2 != w2:
        conf = 0
        mae = 1
    else:
        mae = compute_mae(img1, img2)
        conf = 1 - np.abs((mae - mae_mean) / mae_std)
        conf = bound(conf, 0, 1)

    return conf, mae


def bound(x, a, b):
    x = a if x < a else x
    x = b if x > b else x
    return x


def isNotUnion(r1, r2):
    return False


def compute_mae(img1, img2):
    return np.mean(np.abs(img1 - img2))


def box_dist(res_box, gt_box):
    dist = np.sqrt(
        np.power(res_box[0] - gt_box[0], 2) + np.power(res_box[1] - gt_box[1], 2))
    return dist


def show_res(imgL, roi, res_box, gt_box, save=False):
    roi = list(roi)
    res_box = list(res_box)
    gt_box = list(gt_box)
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(imgL)
    imd = ImageDraw.ImageDraw(im)
    imd.rectangle(roi, outline=(255, 0, 0), width=1)
    imd.text(roi[2:], text='roi')
    imd.rectangle(gt_box, outline=(0, 255, 0), width=1)
    imd.text(gt_box[2:], text='gt_box')
    imd.rectangle(res_box, outline=(0, 0, 255), width=1)
    imd.text(res_box[2:], text='res_box')
    if save:
        im.save(save)
    else:
        im.show()


if __name__ == '__main__':
    from data.dataset import Dataset

    dataset = Dataset.init_dir('../data/face')

    data = dataset[0]
    another_data = dataset[1]

    data.load(trans=False, compress=False)
    another_data.load(trans=False, compress=False)

    box = data.box.copy()
    box[0] -= 50
    box[1] -= 100
    box[2] += 50
    box[3] += 50
    # box = [0, 0, data.imgL.shape[1], data.imgL.shape[0]]
    res = main_search(data.imgL, another_data.imgS, box)
    res['dist'] = box_dist(res['box'], data.box)
    print(res)
    show_res(data.imgL, roi=box, res_box=res['box'], gt_box=data.box)
