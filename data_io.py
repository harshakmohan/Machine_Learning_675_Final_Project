import numpy as np
import os
import cv2 as cv
import random
import pickle

base = "./data"
categories = ['cnv', 'dme', 'drusen', 'normal']

def create_dataset():
    data = []
    for category in categories:
        class_num = categories.index(category)
        path = path.os.join(base, category)
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

    out = open('X.pickle', 'wb')
    pickle.dump(X, out)
    out.close()

    out = open('y.pickle', 'wb')
    pickle.dump(y, out)
    out.close()



 
