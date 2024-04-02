import numpy as np
import matplotlib.pyplot as plt
from typing import List
import torch
from torch import nn, Tensor
import seaborn as sns

# from grassmann import compute_distances_on_grassmann_mdf

def metrics(labels, pred):
    from sklearn.metrics import confusion_matrix, accuracy_score  # f1_score

    assert labels.shape == pred.shape, f'their shape is labels: {labels.shape}, pred:{pred.shape}'

    acc = accuracy_score(labels.numpy(), pred.numpy())
    c = confusion_matrix(
        labels.numpy(), pred.numpy(),
        normalize='false'
    )
    return 100 * acc, c


def winner_prototype_indices(ydata: torch.Tensor, yprotos_mat: torch.Tensor, distances: torch.Tensor):
    """
    Find the closest prototypes to a batch of features
    :param ydata: labels of input images, SHAPE: (batch_size,)
    :param yprotos_mat: labels of prototypes, SHAPE: (nclass, number_of_prototypes)
    Note: we can use it for both prototypes with the same or different labels (W^+ and W^-)
    :param distances: distances between images and prototypes, SHAPE: (batch_size, number_of_prototypes)
    :return: a dictionary containing winner prototypes
    """
    assert distances.ndim == 2, (f"There should be a distance matrix of shape (batch_size, number_of_prototypes), "
                                 f"but it gets {distances.shape}")

    Y = yprotos_mat[ydata]
    distances_sparse = distances * Y
    winners = torch.zeros_like(distances)
    # Note: here we assume that prototypes with the same label are next to each other in the xprotos tensor
    return torch.stack(
        [
            torch.argmin(w[torch.nonzero(w)]) + torch.nonzero(w)[0] for w in torch.unbind(distances_sparse)
        ], dim=0
    ).T


def winner_prototype_distances(
        ydata: torch.Tensor,
        yprotos_matrix: torch.Tensor,
        yprotos_comp_matrix: torch.Tensor,
        distances: torch.Tensor
):
    """
    find the distance between winners' prototypes and data
    :param ydata: a (nbatch,) array containing labels of data
    :param yprotos_matrix: a (nclass, nprotos) matrix containing non-zero elements in c-th row for prototypes with label 'c'
    :param distantces: (nbatch, nprotos) matrix containing distances between data and prototypes
    :return: D^{+,-} matrices of size (nbatch, nprotos) containing zero on not-winner prototypes
    """
    nbatch, nprotos = distances.shape
    iplus = winner_prototype_indices(ydata, yprotos_matrix, distances)[0]
    iminus = winner_prototype_indices(ydata, yprotos_comp_matrix, distances)[0]

    Dplus = torch.zeros_like(distances)
    Dminus = torch.zeros_like(distances)
    Dplus[torch.arange(nbatch), iplus] = distances[torch.arange(nbatch), iplus]
    Dminus[torch.arange(nbatch), iminus] = distances[torch.arange(nbatch), iminus]

    return Dplus, Dminus, iplus, iminus


def MU_fun(Dplus, Dminus):
    """
    Mu = (D^+ - D^-)/(D^++D^-)
    :param ydata: a (nbatch,) array containing labels of data
    :param yprotos_matrix: a (nclass, nprotos) matrix containing non-zero elements in c-th row for prototypes with label 'c'
    :param distantces: (nbatch, nprotos) matrix containing distances between data and prototypes
    :return: an array of size (nbatch,) containing mu values
    """
    return (Dplus - Dminus).sum(axis=1) / (Dplus + Dminus).sum(axis=1)



def activation_function(act_type: str, sigma: float = 100):
    def f(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix: torch.Tensor):
        assert act_type in ['sigmoid',
                            'identity'], (f"'{act_type}' is an invalid function! you can only pick sigmoid and "
                                          f"identity function")

        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix)

        if act_type == 'sigmoid':
            return (nn.Sigmoid()(sigma * MU_fun(Dplus, Dminus)), iplus, iminus)
        elif act_type == 'identity':
            return (MU_fun(Dplus, Dminus), iplus, iminus)

    return f


def activation_function_derivative(act_type: str, sigma: float = 1):
    def f(x: torch.Tensor):
        assert act_type in ['sigmoid',
                            'identity'], (f"'{act_type}' is an invalid function! you can only pick sigmoid and "
                                          f"identity function")

        if act_type == 'sigmoid':
            return sigma * x * (1 - x)
        elif act_type == 'identity':
            return torch.ones_like(x)

    return f


def get_class_weight(labels):
    l = labels.numpy()
    w = l.shape[0] / (np.unique(l).shape[0] * np.bincount(l.astype('int64')))
    return torch.from_numpy(w)



def errorcurves(acc_train: List, lamda: List, acc_val: List = None):
    nepochs = list(range(len(acc_train)))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(nepochs, acc_train, label='train set')
    if acc_val.size is not None:
        ax[0].plot(nepochs, acc_val, label='validation set')
    ax[0].set_title('accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('accuracy')
    ax[0].tick_params(bottom=True, top=False, left=True, right=True)
    plt.legend()

    r = list(range(1, lamda.shape[1] + 1))
    for i, l in enumerate(lamda):
        # ax[1].plot(
        #     r,
        #     l,
        #     label= str(i)
        # )
        sns.barplot(x=r, y=l, palette="Greens", linewidth=1.5, edgecolor=".1", ax=ax[1])
        ax[1].set_title('lambda')
        ax[1].set_xlabel('index of dim')
        ax[1].set_ylabel('importance')
    # plt.legend()
    plt.show()
