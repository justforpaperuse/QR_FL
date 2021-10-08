import math
import numpy as np
from tensorflow.keras import datasets
from tensorflow.python.keras import models
from tensorflow.python.keras.utils.np_utils import to_categorical

from FL_QR_mnist.model import D_CNN


class client_data:
    def __init__(self, train_images, train_labels):
        self.x = train_images
        self.y = train_labels


def Dataset(spilt_num, one_hot=True):
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = train_images / 255.0
    if one_hot:
        train_labels = to_categorical(train_labels, 10)
    random_order = list(range(len(train_images)))
    # np.random.shuffle(random_order)
    yushu = len(train_images) % spilt_num
    if yushu != 0:
        random_order.extend(random_order[:spilt_num - yushu])
    train_images = train_images[random_order]
    train_labels = train_labels[random_order]
    res = []
    for x, y in zip(np.split(train_images, spilt_num), np.split(train_labels, spilt_num)):
        res.append(client_data(x, y))
    return res


class Client:
    def __init__(self, input_shape, clients_num):
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
        self.test_images = test_images
        self.test_labels = test_labels
        self.clients_num = clients_num
        self.model = D_CNN(input_shape)
        self.dataset = Dataset(clients_num)

    def run_test(self, test_num):
        test_images = self.test_images[:test_num]
        test_labels_OneHot = to_categorical(self.test_labels[:test_num], 10)
        test_images=test_images.reshape(-1, 28, 28, 1)
        loss, acc = self.model.evaluate(test_images,
                                        test_labels_OneHot, verbose=0)
        return loss, acc

    def train_epoch(self, client_id):
        train_images_C = self.dataset[client_id].x
        train_labels_OneHot_C = self.dataset[client_id].y
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        train_images_C = train_images_C.reshape(-1, 28, 28, 1)
        train_history_C = self.model.fit(train_images_C, train_labels_OneHot_C,
                                         validation_split=0.2,
                                         epochs=1, batch_size=128, verbose=1)
        return train_history_C.history

    def choose_clients(self, ratio=1.0):
        choose_num = math.ceil(self.clients_num * ratio)
        return np.random.permutation(self.clients_num)[:choose_num]  # 序列进行随机排序
