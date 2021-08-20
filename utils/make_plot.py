import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
import numpy as np

def loss_plot(train_loss, val_loss, epoch, name):

    plt.plot(epoch, val_loss, marker='.', c='red', label="Validation-set Loss")
    plt.plot(epoch, train_loss, marker='.', c='blue', label="Train-set Loss")

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    plt.savefig(f"../img/{name}_loss.png")
    return


def acc_plot(train_acc, val_acc, epoch, name):
    plt.plot(epoch, val_acc, marker='.', c='red', label="Validation-set Loss")
    plt.plot(epoch, train_acc, marker='.', c='blue', label="Train-set Loss")

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    plt.savefig(f"../img/{name}_acc.png")
    return

if __name__== '__mian__':
    print('hi')
    t_l =  np.random.random(10)
    v_l = np.random.random(10)
    epoch = list(range(1,21, 2))
    loss_plot(t_l, v_l, epoch, 'temp')
    # acc_plot()


