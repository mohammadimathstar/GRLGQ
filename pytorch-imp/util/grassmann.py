import numpy as np
from scipy import linalg as LA
from typing import List
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns

# SIGMOID = nn.Sigmoid()


def grassmann_repr(batch_imgs: torch.Tensor, dim_of_subspace: int) -> torch.Tensor:
    """

    :param batch_imgs: a batch of features of size (batch size, num_of_channels, W, H)
    :param dim_of_subspace: the dimensionality of the extracted subspace
    :return: an orthonormal matrix of size (batch size, W*H, dim_of_subspace)
    """
    assert batch_imgs.ndim == 4, f"xs should be of the shape (batch_size, nchannel, w, h), but it is {batch_imgs.shape}"

    bsize, nchannel, w, h = batch_imgs.shape
    xs = torch.transpose(batch_imgs.view(bsize, nchannel, w * h), 1, 2)

    # SVD: generate principal directions
    U, S, Vh = torch.linalg.svd(
        xs,
        full_matrices=False,
        # driver='gesvd',
    )
    return U[:, :, :dim_of_subspace]


def init_randn(
        dim_of_data: int,
        dim_of_subspace: int,
        labels: torch.Tensor = None,
        num_of_protos: [int, torch.Tensor] = 1,
        num_of_classes: int = None,
):
    """ Initialize prototypes randomly using a Gaussian distribution."""
    if labels is None:
        classes = torch.arange(num_of_classes)
    else:
        classes = torch.unique(labels)

    if isinstance(num_of_protos, int):
        total_num_of_protos = len(classes) * num_of_protos
    else:
        total_num_of_protos = torch.sum(num_of_protos).item()

    nclass = len(classes)
    prototype_shape = (total_num_of_protos, dim_of_data, dim_of_subspace)

    xprotos = np.random.normal(0, 1, size=prototype_shape)
    yprotos = torch.from_numpy(np.repeat(classes.numpy(), num_of_protos)).to(torch.int32)
    yprotos_mat = torch.zeros((nclass, total_num_of_protos), dtype=torch.int32)
    yprotos_mat_comp = torch.zeros((nclass, total_num_of_protos), dtype=torch.int32)

    # orthonormalize prototypes
    for i, proto in enumerate(xprotos):
        xprotos[i] = LA.orth(proto)
        yprotos_mat[yprotos[i], i] = 1
        tmp = list(range(len(classes)))
        tmp.pop(yprotos[i])
        yprotos_mat_comp[tmp, i] = 1

    xprotos = nn.Parameter(torch.from_numpy(xprotos))

    return xprotos, yprotos, yprotos_mat, yprotos_mat_comp


def init_with_samples(
        data: np.array,
        labels: [List, np.array],
        num_of_protos: [int, List]
):
    """ initialize prototypes with data points (on the Grassmann manifold) """

    data = data.numpy()
    labels = labels.numpy()

    eps = 0.0001
    classes = np.unique(labels)
    _, dim_of_data, dim_of_subspace = data.shape

    if isinstance(num_of_protos, int):
        total_num_of_protos = len(classes) * num_of_protos
        num_of_protos = np.repeat(num_of_protos, len(classes))
    else:
        total_num_of_protos = np.sum(num_of_protos)

    xprotos = np.zeros(
        (total_num_of_protos, dim_of_data, dim_of_subspace))
    t = 0
    for c, n in zip(classes, num_of_protos):
        idx = np.argwhere(labels == c).T[0]

        selected_idx = np.random.choice(idx, n, replace=False)
        tmp = data[selected_idx] + eps * np.random.randn(n, dim_of_data, dim_of_subspace)
        xprotos[t: t + n] = np.array([LA.orth(sample) for sample in tmp])
        t += n

    yprotos = np.repeat(classes, num_of_protos)

    return torch.from_numpy(xprotos), torch.from_numpy(yprotos)


def prediction(data, xprotos, yprotos, lamda, metric_type):
    results = compute_distances_on_grassmann_mdf(
        data, xprotos,
        metric_type=metric_type,
        relevance=lamda
    )
    return yprotos[results['distance'].argmin(axis=1)]



def relevances_grad(metric_type: str):
    """
    Compute the (Euclidean) derivative of the distance with respect to the relevance factors.
    """

    def f(canonicalcorrelation):
        assert metric_type in ['geodesic',
                               'chordal',
                               'pseudo-chordal'], \
            f"'{metric_type}' is an invalid distance! you can only pick from ('geodesic', 'chordal', pseudo-chordal'."
        if metric_type == 'pseudo-chordal':
            return canonicalcorrelation
        else:
            return torch.acos(canonicalcorrelation) ** 2

    return f


def distance_grad(proto_type: str):
    """
    \partial{mu} / \partial{d^{+-}}
    :param proto_type:
    :return:
    """
    def f(dplus, dminus):
        if proto_type == 'plus':
            return 2 * dminus / ((dplus + dminus) ** 2)
        else:
            return -2 * dplus / ((dplus + dminus) ** 2)

    return f


def prototype_grad(
        metric_type: str):
    """
    Compute the derivative of the distance with respect to W^{+-} (principal vectors).
    """

    # def f(X_rotated, relevances, canonicalcorrelation=None, D: int=None):
    def f(X_rotated, relevances, **kwargs):
        assert metric_type in ['geodesic',
                               'chordal',
                               'pseudo-chordal'], \
            f"'{metric_type}' is an invalid distance! you can only pick from ('geodesic', 'chordal', pseudo-chordal'."
        if metric_type == 'pseudo-chordal':
            D = kwargs['dim_of_data']
            Lam = torch.tile(
                relevances[0],
                (D, 1)
            )
            return - Lam * X_rotated
        else:
            canonicalcorrelation = kwargs['canonicalcorrelation']
            G = 2 * torch.diag(
                relevances[0] * torch.acos(canonicalcorrelation) / torch.sqrt(1 - canonicalcorrelation ** 2)
            )
            return -X_rotated @ G

    return f




def compute_distances_on_grassmann_mdf(
        xdata: torch.Tensor,
        xprotos: torch.Tensor,
        metric_type: str = 'pseudo-chordal',
        relevance: np.array = None
):
    """
    Compute the (geodesic or chordal) distances between an input subspace and all prototypes.
    """
    assert xdata.ndim == 3, f"xs should be of the shape (batch_size, W*H, dim_of_subspace), but it is {xdata.shape}"

    if relevance is None:
        relevance = torch.ones((1, xprotos.shape[-1])) / xprotos.shape[-1]
    xdata = xdata.unsqueeze(dim=1)

    U, S, Vh = torch.linalg.svd(
        torch.transpose(xdata, 2, 3) @ xprotos.to(xdata.dtype),
        full_matrices=False,
        # driver='gesvd',
    )

    if metric_type == 'pseudo-chordal':
        distance = 1 - torch.transpose(
            relevance @ torch.transpose(S, 1, 2),
            1, 2
        )
    else:
        distance = torch.transpose(
            relevance @ torch.transpose(
                torch.acos(S) ** 2,
                1, 2
            ),
            1, 2
        )

    if torch.isnan(distance).any():
        raise Exception('Error: NaN values! Using the --log_probabilities flag might fix this issue')

    output = {
        'Q': U, # SHAPE: (batch_size, num_of_prototypes, WxH, dim_of_subspaces)
        'Qw': torch.transpose(Vh, 2, 3), # SHAPE: (batch_size, num_of_prototypes, WxH, dim_of_subspaces)
        'canonicalcorrelation': S, # SHAPE: (batch_size, num_of_prototypes, dim_of_subspaces)
        'distance': torch.squeeze(distance, -1), # SHAPE: (batch_size, num_of_prototypes)
    }
    return output


# def winner_prototype_indices(ydata: torch.Tensor, yprotos_mat: torch.Tensor, distances: torch.Tensor):
#     """
#     Find the closest prototypes to a batch of features
#     :param ydata: labels of input images, SHAPE: (batch_size,)
#     :param yprotos_mat: labels of prototypes, SHAPE: (nclass, number_of_prototypes)
#     Note: we can use it for both prototypes with the same or different labels (W^+ and W^-)
#     :param distances: distances between images and prototypes, SHAPE: (batch_size, number_of_prototypes)
#     :return: a dictionary containing winner prototypes
#     """
#     assert distances.ndim == 2, (f"There should be a distance matrix of shape (batch_size, number_of_prototypes), "
#                                  f"but it gets {distances.shape}")
#
#     Y = yprotos_mat[ydata]
#     distances_sparse = distances * Y
#     # Note: here we assume that prototypes with the same label are next to each other in the xprotos tensor
#     return torch.stack(
#         [
#             torch.argmin(w[torch.nonzero(w)]) + torch.nonzero(w)[0] for w in torch.unbind(distances_sparse)
#         ], dim=0
#     ).T
#
#
# def winner_prototypes(results: dict, ydata, yprotos_mat, yprotos_comp_mat):
#     distances = results['distance']
#     iplus = winner_prototype_indices(ydata, yprotos_mat, distances)
#     iminus = winner_prototype_indices(ydata, yprotos_comp_mat, distances)
#
#     dplus = distances[:, iplus]
#     dminus = distances[:, iminus]
#
#     plus = {
#         'index': iplus,
#         'distance': dplus,
#         'Q': results['Q'][:, iplus],
#         'Qw': results['Qw'][:, iplus],
#         'canonicalcorr': results['canonicalcorrelation'][:, iplus]
#     }
#
#     minus = {
#         'index': iminus,
#         'distance': dminus,
#         'Q': results['Q'][:, iminus],
#         'Qw': results['Qw'][:, iminus],
#         'canonicalcorr': results['canonicalcorrelation'][:, iminus]
#     }
#     return plus, minus



if __name__ == '__main__':
    x = torch.randn((1, 4))
    y = torch.Tensor([0, 0, 1, 1])
    yp = torch.Tensor([0, 0, 1, 1])
    # nearest_prototypes(y, yp, x)
    # x = np.random.randn(4, 2)
    # xx = np.repeat(np.expand_dims(x, axis=0), 3, axis=0)
    # # print(x.shape, xx.shape, np.transpose(xx, (0, 2, 1)).shape)
    # xxx = np.expand_dims(np.transpose(np.expand_dims(x, axis=0), (0, 2, 1)), axis=0) @ xx
    # print(x.shape, xx.shape, xxx.shape)
    # U, S, Vh = np.linalg.svd(x.T @ x, full_matrices=False, compute_uv=True, hermitian=False)
    # U2, S2, Vh2 = np.linalg.svd(xxx, full_matrices=False, compute_uv=True, hermitian=False)
    #
    # # print('x x', U)
    # # print('xxx', U2)
    # # print('x x', Vh)
    # # print('xxx', Vh2)
    # print('x x', S)
    # print('xxx', S2)
    # print(U.shape, U2.shape)
    # print(S2.shape)
    # print(np.squeeze(S2).shape)
    # y = np.random.randn(1, 1, 2, 2, 1)
    # print(np.squeeze(y, axis=2).shape)
    # # print(U.ndim, U2.ndim)
    # # print(np.expand_dims(x, axis=(0,1)).shape)
