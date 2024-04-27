import torch
from torch import nn, Tensor
import argparse

from sklearn.metrics import confusion_matrix, accuracy_score  # f1_score


def metrics(y_true: Tensor, y_pred: Tensor, nclasses):

    assert y_true.shape == y_pred.shape, f'their shape is labels: {y_true.shape}, pred:{y_pred.shape}'

    acc = accuracy_score(y_true.numpy(), y_pred.numpy())
    c = confusion_matrix(
        y_true.numpy(), y_pred.numpy(),
        labels=range(nclasses),
        # normalize='true'
    )
    return acc, c


def winner_prototype_indices(ydata: Tensor, yprotos_mat: Tensor, distances: Tensor):
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

    # print(ydata.get_device(), yprotos_mat.get_device())
    Y = yprotos_mat[ydata.tolist()]
    distances_sparse = distances * Y

    return torch.stack(
        [
            torch.argwhere(w).T[0,
            torch.argmin(
                w[torch.argwhere(w).T],
            )
            ] for w in torch.unbind(distances_sparse)
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
    iplus = winner_prototype_indices(ydata, yprotos_matrix, distances)
    iminus = winner_prototype_indices(ydata, yprotos_comp_matrix, distances)
    # print('winner_prototype_distance')
    # print(iplus)

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


def IdentityLoss():
    def f(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix: torch.Tensor):
        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix)
        return (nn.Identity()(MU_fun(Dplus, Dminus)).sum(), iplus, iminus)
    return f


def SigmoidLoss(sigma: int=100):
    def f(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix: torch.Tensor):
        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix)
        return (nn.Sigmoid()(sigma * MU_fun(Dplus, Dminus)).sum(), iplus, iminus)
    return f


def ReLULoss():
    def f(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix: torch.Tensor):
        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix)
        return (nn.ReLU()(MU_fun(Dplus, Dminus)).sum(), iplus, iminus)
    return f


def LeakyReLULoss(negative_slope: float=0.01):
    def f(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix: torch.Tensor):

        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix)

        # print('leakyrelu', ydata.shape, Dplus.shape, iplus)
        return (nn.LeakyReLU(negative_slope)(MU_fun(Dplus, Dminus)).sum(), iplus, iminus)
    return f


def ELULoss(alpha: float = 1):
    def f(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix: torch.Tensor):
        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix)
        return (nn.ELU(alpha)(MU_fun(Dplus, Dminus)).sum(), iplus, iminus)
    return f


def RReLULoss(lower=0.125, upper=0.3333333333333333):
    def f(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix: torch.Tensor):
        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix)
        return (nn.RReLU(lower, upper)(MU_fun(Dplus, Dminus)).sum(), iplus, iminus)
    return f


def get_loss_function(args: argparse.Namespace):
    if args.cost_fun == 'sigmoid':
        sigma = args.sigma or 100
        return SigmoidLoss(sigma)
    elif args.cost_fun == 'relu':
        return ReLULoss()
    elif args.cost_fun == 'leaky_relu':
        sigma = args.sigma or 0.1
        return LeakyReLULoss(sigma)
    elif args.cost_fun == 'elu':
        sigma = args.sigma or 1
        return ELULoss(alpha=sigma)
    elif args.cost_fun == 'rrelu':
        return RReLULoss()
    else:
        return IdentityLoss()