import numpy as np
from tensorflow.python.keras.models import load_model
from tqdm import tqdm
# from FL_QR_mnist.avg_Client import Client
from FL_QR_mnist.avg_Client_Q import Client
###########################    avg train   ########################

clients_num = 100
CLIENT_RATIO_PER_ROUND = 0.12
epoch = 100
client = Client((28, 28, 1), clients_num)

history = client.train_epoch(0)
global_vars = client.model.get_weights()
 # client.model.save("FL_cifar//result//global_model.h5")

loss_all = []
acc_all = []
for ep in range(epoch):
    client_vars_sum = None
    random_clients = client.choose_clients(CLIENT_RATIO_PER_ROUND)
    for client_id in tqdm(random_clients, ascii=True):
        client.model.set_weights(global_vars)
        client.train_epoch(client_id)
        current_client_vars = client.model.get_weights()
        # sum it up
        if client_vars_sum is None:
            client_vars_sum = current_client_vars
        else:
            for cv, ccv in zip(client_vars_sum, current_client_vars):
                cv += ccv
    # obtain the avg vars as global vars
    global_vars = []
    for var in client_vars_sum:
        global_vars.append(var / len(random_clients))

    client.model.set_weights(global_vars)
    loss, acc = client.run_test(10000)
    print("[epoch {}, {} inst] Testing ACC: {:.4f}, Loss: {:.4f}".format(ep + 1, 600, acc, loss))

    loss_all.append(loss)
    acc_all.append(acc)

loss_acc = np.concatenate(
    (np.array(loss_all).reshape(len(loss_all), 1), np.array(acc_all).reshape(len(acc_all), 1)), axis=1)
np.save("FL_QR_mnist//result//avg_Q_loss_acc.npy", loss_acc)
client.model.save("FL_QR_mnist//result//avg_Q.h5")



client.model.set_weights(global_vars)
loss, acc = client.run_test(10000)

loss_acc = np.concatenate(
    (np.array(loss_all).reshape(len(loss_all), 1), np.array(acc_all).reshape(len(acc_all), 1)), axis=1)
np.save("FL_QR_mnist//result//avg_loss_acc.npy", loss_acc)
client.model.save("FL_QR_mnist//result//avg.h5")
###########################    avg Q  train   ########################
np.save("FL_QR_mnist//result//avg_Q_loss_acc.npy", loss_acc)
client.model.save("FL_QR_mnist//result//avg_Q.h5")

###########################    avg R  train   ########################
np.save("FL_QR_mnist//result//avg_R_loss_acc.npy", loss_acc)
client.model.save("FL_QR_mnist//result//avg_R.h5")


###########   z a ############
loss_acc = np.load("FL_QR_mnist//result//avg_R_loss_acc.npy")
model = load_model("FL_QR_mnist//result//avg_R.h5")
loss_all= loss_acc[:, 0].tolist()
acc_all = loss_acc[:, 1].tolist()
