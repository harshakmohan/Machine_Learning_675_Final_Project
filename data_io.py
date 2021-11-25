import numpy as np
import os
import cv2 as cv 

base = "./data"
categories = ['cnv', 'dme', 'drusen', 'normal']
def load():
    data = []
    for category in categories:
        class_num = categories.index(category)
        path = path.os.join(base, category)
        for img in os.listdir(path):
            img_arr = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)
        data.append([img_arr, class_num])


