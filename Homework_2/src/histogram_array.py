import numpy as np


class HistogramArray:
    def __init__(self, window):
        self.bincounts = np.zeros(256, dtype=int)
        self.center = len(window) // 2

        counts = np.bincount(window)
        self.bincounts[:len(counts)] += counts

    def add(self, pixels):
        counts = np.bincount(pixels)
        self.bincounts[:len(counts)] += counts

    def delete(self, pixels):
        counts = np.bincount(pixels)
        self.bincounts[:len(counts)] -= counts

    def median(self):
        return np.where(np.cumsum(self.bincounts) > self.center)[0][0]
