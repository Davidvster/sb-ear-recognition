import abc
import os
import random
from abc import abstractmethod

import numpy
from keras_preprocessing.image import load_img, img_to_array
from skimage import io
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import np_utils

import constant
from common import Common


FILE_TRAIN_CSV = "awe-train.csv"

class EarDataSet(metaclass=abc.ABCMeta):

    def __init__(self, n_classes):
        # to je steilo oseb - pri AWE je 100
        self.n_classes = n_classes
        # objects so podatki slik, ki so prebrani z io.imread() in potem reshapani ampak bo jih treba resizat tudi, tako da uporabi
        #                     image = load_img(img_abs_path, target_size=(constant.IMG_WIDTH, constant.IMG_HEIGHT))
        #                     image = img_to_array(image)
        #                     image = Common.reshape_from_img(image)
        #                     objects.append(image)
        #                  objects = Common.to_float(numpy.asarray(self.objects, dtype=numpy.float32))
        self.objects = []
        # labels so train data in so vektorji dolgi 100, ki imajo povsod 0, razen pri osebi ki je ta prava imajo 1
        self.labels = []
        # obj_validation so podatki slik, ki so prebrani z io.imread() in potem reshapani ampak bo jih treba resizat tudi, tako da uporabi
        #                     image = load_img(img_abs_path, target_size=(constant.IMG_WIDTH, constant.IMG_HEIGHT))
        #                     image = img_to_array(image)
        #                     image = Common.reshape_from_img(image)
        #                     obj_validation.append(image)
        #                  obj_validation = Common.to_float(numpy.asarray(self.obj_validation, dtype=numpy.float32))
        self.obj_validation = []
        # labels_validation so test data in so vektorji dolgi 100, ki imajo povsod 0, razen pri osebi ki je ta prava imajo 1
        self.labels_validation = []
        # number_labels je steilo vseh slik, ki sodelujejo v treniranju CNN modela
        self.number_labels = 0

    def get_data(self):
        self.objects, self.labels = self.fetch_img_path()
        self.process_data()
        self.print_dataSet()

    def split_training_set(self):
        return train_test_split(self.objects, self.labels, test_size=0.3,
                                random_state=random.randint(0, 100))

    def process_data(self):
        self.objects, self.img_obj_validation, self.labels, self.img_labels_validation = self.split_training_set()
        self.labels = np_utils.to_categorical(self.labels, self.n_classes)
        self.labels_validation = np_utils.to_categorical(self.img_labels_validation, self.n_classes)
        self.obj_validation = Common.to_float(numpy.asarray(self.obj_validation, dtype= numpy.float32))
        self.objects = Common.to_float(numpy.asarray(self.objects, dtype= numpy.float32))

    def fetch_img_path(self):
        images = []
        labels = []
        for line in open(FILE_TRAIN_CSV):
            csv_row = line.split(",")  # returns a list ["1","50","60"]
            file_name = csv_row[1]
            image = load_img("awe/"+file_name, target_size=(constant.IMG_WIDTH,constant.IMG_HEIGHT))
            image = img_to_array(image)
            image = Common.reshape_from_img(image)
            label = int(csv_row[2])-1
            images.append(image)
            labels.append(label)
            self.number_labels +=1

        return images, labels

    def print_dataSet(self):
        print(self.objects)
        print(self.labels)
