import time
import numpy as np


class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count


class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider


def calculate_psnr(img1, img2):
    img1 = np.asarray(img1, dtype=float)
    img2 = np.asarray(img2, dtype=float)
    mse = np.average((img1 - img2) * (img1 - img2))
    psnr = 20 * np.log10(255.) - 10 * np.log10(mse)
    return psnr

