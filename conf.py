import numpy as np


def compute_conf(data, thresh):
    return 1 - data / thresh / 3


if __name__ == '__main__':
    data = 0.01
    thresh = 0.05
    print(compute_conf(data, thresh))
