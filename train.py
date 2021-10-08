import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical

from FL_QR_mnist.model import DL_CNN, D_CNN
from tensorflow.keras import datasets

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images_Q = np.load("FL_QR_mnist//data//train_Q.npy")  # "solve_DL_gradient//train_gam_32.npy"
test_images_Q = np.load("FL_QR_mnist//data//test_Q.npy")  # "solve_DL_gradient//test_gam_32.npy"
train_images_R = np.load("FL_QR_mnist//data//train_R.npy")
test_images_R = np.load("FL_QR_mnist//data//test_R.npy")

train_labels_OneHot = to_categorical(train_labels)
test_labels_OneHot = to_categorical(test_labels)


############################# all
model = D_CNN((28, 28, 1))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
train_images = train_images[:500].reshape(-1, 28, 28, 1)
model.fit(train_images, train_labels_OneHot[:500], batch_size=128, epochs=20, validation_split=0.2)

test_images = test_images.reshape(10000, 28, 28, 1)
score = model.evaluate(test_images, test_labels_OneHot)
model.save("za_all//all_local.h5")
#########################    Q_CNN   ###############
model = D_CNN((28, 28, 1))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
train_images_Q = train_images_Q.reshape(60000, 28, 28, 1)
loss_acc = []
loss_acc0 = []
loss_acc1 = []

model.fit(train_images_Q, train_labels_OneHot, batch_size=128, epochs=30, validation_split=0.2)
loss, acc = model.history.history['loss'], model.history.history['accuracy']
val_accuracy = model.history.history['val_accuracy']
np.save("FL_QR_mnist//result// Q_val_accuracy.npy", val_accuracy)

loss_acc0.extend(loss)
loss_acc1.extend(acc)

loss_acc = np.concatenate(
    (np.array(loss_acc0).reshape(len(loss_acc0), 1), np.array(loss_acc1).reshape(len(loss_acc1), 1)), axis=1)

np.save("FL_QR_mnist//result//Q_loss_acc.npy", loss_acc)

test_images_Q = test_images_Q.reshape(10000, 28, 28, 1)
score = model.evaluate(test_images_Q, test_labels_OneHot)

############################   R_CNN    #################

model = D_CNN((28, 28, 1))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

loss_acc0 = []
loss_acc1 = []
train_images_R = train_images_R.reshape(60000, 28, 28, 1)

model.fit(train_images_R , train_labels_OneHot, batch_size=128, epochs=30, validation_split=0.2)
loss, acc = model.history.history['loss'], model.history.history['accuracy']
val_accuracy = model.history.history['val_accuracy']
np.save("FL_QR_mnist//result//R_val_accuracy.npy", val_accuracy)

loss_acc0.extend(loss)
loss_acc1.extend(acc)

loss_acc = np.concatenate((np.array(loss_acc0).reshape(len(loss_acc0), 1), np.array(loss_acc1).reshape(len(loss_acc1),1)), axis=1)
np.save("FL_QR_mnist//result//R_loss_acc.npy.npy", loss_acc)

test_images_R = test_images_R.reshape(10000, 28, 28, 1)
score = model.evaluate(test_images_R, test_labels_OneHot)

########################    QR     #########################
model = DL_CNN()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
train_images_Q = train_images_Q.reshape(60000, 28, 28, 1)
train_images_R = train_images_R.reshape(60000, 28, 28, 1)

# loss_acc = namedtuple('loss_acc', ['loss', 'accuracy'])
loss_acc0 = []
loss_acc1 = []

model.fit([train_images_Q, train_images_R], train_labels_OneHot, batch_size=128, epochs=10)
loss, acc = model.history.history['loss'], model.history.history['accuracy']
loss_acc0.extend(loss)
loss_acc1.extend(acc)

loss_acc = np.concatenate((np.array(loss_acc0).reshape(len(loss_acc0), 1), np.array(loss_acc1).reshape(len(loss_acc1),1)), axis=1)
np.save("FL_QR_mnist//result//QR_loss_acc.npy", loss_acc)

test_images_Q = test_images_Q.reshape(10000, 28, 28, 1)
test_images_R = test_images_R.reshape(10000, 28, 28, 1)
score = model.evaluate([test_images_Q, test_images_R], test_labels_OneHot)


################# corr ##################
np.corrcoef(train_images_Q[0].reshape(1,-1),train_images_Q[2].reshape(1,-1))
np.corrcoef(train_images_R[0].reshape(1,-1),train_images_R[2].reshape(1,-1))
np.corrcoef(train_images_Q[0].reshape(1,-1),train_images_Q[-3].reshape(1,-1))