import numpy as np
import os
import cv2 as cv
import random
from sklearn.utils import shuffle


class DataIO():

    def __init__(self, dir="./dev", grayscale=True, size=100):
        self.base = dir
        self.dataset_size = size
        self.categories = {'CNV': 1, 'DME': 2, 'DRUSEN': 3, 'NORMAL': 0}

    def create_dataset(self, out_X='X', out_y='y', func=None): # func will be preprocessing function
        X = []
        y = []
        for category in self.categories:
            class_num = self.categories[category]*np.ones((30,30))
            path = os.path.join(self.base, category)
            for img in random.choices(os.listdir(path), k=self.dataset_size):
                img_arr = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)
                img_arr = cv.resize(img_arr, (496,496))
                if func:
                    img_arr = func(img_arr)
                X.append(img_arr)
                y.append(class_num)

        X, y = np.array(X), np.array(y)
        X, y = shuffle(X, y, random_state=0)

        np.save(out_X, X)
        np.save(out_y, y)

if __name__ == '__main__':
    DataIO().create_dataset()