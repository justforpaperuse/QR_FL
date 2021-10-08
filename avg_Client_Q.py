import math
import numpy as np
from tensorflow.keras import datasets
from tensorflow.python.keras import models
from tensorflow.python.keras.utils.np_utils import to_categorical

from FL_QR_mnist.model import D_CNN


def train_x(spilt_num):
    train_images = np.load("FL_QR_mnist//data//train_Q.npy")
    train_images = train_images.reshape(60000, 28, 28, 1)
    k = int(len(train_images) / spilt_num)
    res = []
    start = 0
    for i in range(spilt_num):
        x = train_images[start:start+k]
        res.append(x)
        start = start + k
    return res


def train_y(spilt_num):
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_labels_OneHot = to_categorical(train_labels)
    k = int(len(train_images) / spilt_num)
    res = []
    start = 0
    for i in range(spilt_num):
        x = train_labels_OneHot[start:start+k]
        res.append(x)
        start = start + k
    return res


class Client:
    def __init__(self, input_shape, clients_num):
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
        self.test_images = np.load("FL_QR_mnist//data//test_Q.npy")
        self.test_labels = test_labels
        self.clients_num = clients_num
        self.model = D_CNN(input_shape)
        self.train_images = train_x(clients_num)
        self.train_labels = train_y(clients_num)

    def run_test(self, test_num):
        test_images = self.test_images[:test_num]
        test_labels_OneHot = to_categorical(self.test_labels[:test_num], 10)
        test_images = test_images.reshape(-1, 28, 28, 1)
        loss, acc = self.model.evaluate(test_images,
                                        test_labels_OneHot, verbose=0)
        return loss, acc

    def train_epoch(self, client_id):
        train_images_C = self.train_images[client_id]
        train_labels_OneHot_C = self.train_labels[client_id]
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        train_history_C = self.model.fit(train_images_C, train_labels_OneHot_C, epochs=1, batch_size=128, validation_split=0.2)
        return train_history_C.history

    def choose_clients(self, ratio=1.0):
        choose_num = math.ceil(self.clients_num * ratio)
        return np.random.permutation(self.clients_num)[:choose_num]  # 序列进行随机排序
