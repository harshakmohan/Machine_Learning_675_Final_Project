import numpy as np
import os
import cv2 as cv
import random
import pickle

base = "./data/test_train"
class DataLoader():

    def __init__(self, dir="./data/test_train", grayscale=True, size=100):
        self.base = dir
        self.size = size
        self.categories = {'CNV': 1, 'DME': 2, 'DRUSEN': 3, 'NORMAL': 0}

    def create_dataset(self, func): # func will be preprocessing function
        data = []
        for category in self.categories:
            class_num = self.categories[category]
            path = os.path.join(base, category)
            for img in os.listdir(path):
                img_arr = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)
                data.append([img_arr, class_num])
        random.shuffle(data)
        X = []
        y = []
        for image, label in data:
            X.append(image)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        out = open('X.npy', 'wb')
        pickle.dump(X, out)
        out.close()

        out = open('y.npy', 'wb')
        pickle.dump(y, out)
        out.close()

if __name__ == '__main__':
