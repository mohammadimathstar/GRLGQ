import numpy as np
from scipy import linalg as LA
from typing import List
import matplotlib.pyplot as plt


def init_randn(
        dim_of_data: int,
        dim_of_subspace: int,
        labels: [List, np.array],
        num_of_protos: [int, List] = 1
):
    """ Initialize prototypes randomly using a Gaussian distribution."""

    classes = np.unique(labels)
    if isinstance(num_of_protos, int):
        total_num_of_protos = len(classes) * num_of_protos
    else:
        total_num_of_protos = np.sum(num_of_protos)

    xprotos = np.random.normal(
        0,
        1,
        (total_num_of_protos, dim_of_data, dim_of_subspace)
    )
    # orthonormalize prototypes
    for i, proto in enumerate(xprotos):
        xprotos[i] = LA.orth(proto)

    yprotos = np.repeat(classes, num_of_protos)

    return xprotos, yprotos


def init_with_samples(
        data: np.array,
        labels: [List, np.array],
        num_of_protos: [int, List]
):
    """ initialize prototypes with data points (on the Grassmann manifold) """

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

    return xprotos, yprotos



def sigmoid(x, sigma: int = 100):
    return 1 / (1 + np.exp(- sigma * x))


def loss(d_plus, d_minus, act_fun: str = 'sigmoid', sigma: int = 100):
    """ Calculate the loss value for a data point. """

    if act_fun == 'identity':
        return (d_plus - d_minus) / (d_plus + d_minus)
    else:
        return sigmoid((d_plus - d_minus) / (d_plus + d_minus), sigma)


def compute_distances_on_grassmann_mdf(
        xdata,
        xprotos,
        metric_type: str = 'geodesic',
        relevance: np.array = None
):
    """
    Compute the (geodesic or chordal) distances between an input subspace and all prototypes.
    """
    assert xdata.shape[-2:] == xdata.shape[
                               -2:], f"The size of input data should be {xprotos.shape[1]}, but it is {xdata.shape[0]}!"

    if xdata.ndim == 2:
        xdata = np.expand_dims(xdata, axis=(0, 1))
    elif xdata.ndim == 3:
        xdata = np.expand_dims(xdata, axis=1)

    U, S, Vh = np.linalg.svd(np.transpose(xdata, (0, 1, 3, 2)) @ xprotos, full_matrices=False, compute_uv=True, hermitian=False)

    if metric_type == 'chordal':
        dis = 1 - np.transpose(relevance @ np.transpose(S ** 2, (0, 2, 1)), (0, 2, 1))
    elif metric_type == 'pseudo-chordal':
        dis = 1 - np.transpose(relevance @ np.transpose(S, (0, 2, 1)), (0, 2, 1))
    else:
        dis = np.transpose(relevance @ np.transpose(np.arccos(S) ** 2, (0, 2, 1)), (0, 2, 1))
    if relevance.shape[0] != 1:
        # if it is localized
        dis = np.array([np.diag(d) for d in dis])

    output = {
        'Q': np.squeeze(U),
        'Qw': np.squeeze(np.transpose(Vh, (0, 1, 3, 2))),
        'canonicalcorrelation': np.squeeze(S),
        'distance': np.squeeze(dis),
    }

    assert np.sum(output['distance'] < 0) < 1, "Distance is negative!"
    return output

def nearest_prototypes_ids(distances, label, yprotos):
    sameclass = np.argwhere(yprotos == label).T[0]
    diffclass = np.argwhere(yprotos != label).T[0]

    iplus = sameclass[np.argmin(distances[sameclass])]
    iminus = diffclass[np.argmin(distances[diffclass])]
    return iplus, iminus

def nearest_prototypes(data, label, xprotos, yprotos, metric_type, relevance):
    """ Find the closest prototypes to a given data point """

    results = compute_distances_on_grassmann_mdf(data, xprotos, metric_type, relevance)

    iplus, iminus = nearest_prototypes_ids(results['distance'], label, yprotos)
    distances = results['distance']
    dplus = distances[iplus]
    dminus = distances[iminus]

    plus = {
        'index': iplus,
        'distance': dplus,
        'Q': results['Q'][iplus],
        'Qw': results['Qw'][iplus],
        'canonicalcorr': results['canonicalcorrelation'][iplus]
    }

    minus = {
        'index': iminus,
        'distance': dminus,
        'Q': results['Q'][iminus],
        'Qw': results['Qw'][iminus],
        'canonicalcorr': results['canonicalcorrelation'][iminus]
    }

    return plus, minus


def Deri_act_fun(cost, **kwargs):
    """
    Compute the derivative of the activation function with respect to cost.
    """
    if 'act_fun' in kwargs.keys():
        act_fun = kwargs['act_fun']
    else:
        act_fun = 'sigmoid'

    if act_fun == 'identity':
        return 1
    else:
        if 'sigma' in kwargs.keys():
            sigma = kwargs['sigma']
        else:
            sigma = 100
        return sigma * cost * (1 - cost)


def Deri_cost_dplus(cost, dplus, dminus, **kwargs):
    """
    Compute the derivative of the error function with respect to the distance to W^+ (winner prototype with the same label).
    """

    return 2 * Deri_act_fun(cost, **kwargs) * dminus / ((dplus + dminus) ** 2)

def dE_distance_minus(self, cost, dplus, dminus):
    """
    Compute the derivative of the error function with respect to the distance to W^- (winner prototype with a different label).

    Args:
        cost (float): The amount of cost caused by the sample.
        dplus (float): The distance between the sample and W^+.
        dminus (float): The distance between the sample and W^-.

    Returns:
        float: The derivative of the error function with respect to dminus.

    """

    return -2 * self.der_act_fun(cost) * dplus / ((dplus + dminus) ** 2)


def metrics(targets, pred):
    from sklearn.metrics import confusion_matrix, accuracy_score# f1_score

    assert targets.shape == pred.shape, f'their shape is labels: {targets.shape}, pred:{pred.shape}'

    acc = accuracy_score(targets, pred)
    conf = confusion_matrix(
        targets, pred,
        normalize='true'
    )

    return 100 * acc, 100 * conf

def errorcurves(acc_train: List, lamda: List, acc_val: List=None):
    import seaborn as sns
    import matplotlib.gridspec as gridspec

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

    l = list(range(1, lamda.shape[1] + 1))
    sns.barplot(x=l, y=lamda[0], palette="Greens", edgecolor=".1", ax=ax[1])
    # r = list(range(lamda.shape[1]))
    # for i, l in enumerate(lamda):
    #     ax[1].plot(
    #         r,
    #         l,
    #         label= str(i)
    #     )
    #     ax[1].set_title('lambda')
    #     ax[1].set_xlabel('index of dim')
    #     ax[1].set_ylabel('importance')
    # plt.legend()
    plt.show()

def plot_relevances(relevances, isSaved=False, filename=""):
    l = list(range(1, relevances.shape[0] + 1))
    fig = plt.figure(figsize=(3, 2))  # , constrained_layout=True)
    gs = gridspec.GridSpec(1, 1, figure=fig)
    gs.update(wspace=0.25)
    gs.update(hspace=0.5)

    ax3 = plt.subplot(gs[0])
    sns.barplot(x=l, y=relevances, palette="Greens", edgecolor=".1", ax=ax3)

    ax3.set_xlabel("index of relevance factors", fontweight="bold", fontsize=9)
    ax3.set_ylabel("relevance factors", fontweight="bold", fontsize=9)
    # ax3.set_xticks(range(4, 21, 5))
    ax3.set_yticks([0, .1, .2])

    plt.subplots_adjust(left=0.18,
                        bottom=0.22,
                        right=0.98,
                        top=0.9,
                        wspace=0.05,
                        hspace=0.05)

    if isSaved:
        plt.savefig(filename + ".eps", format="eps", dpi=150)
    plt.show()


def return_model(fname):
    with np.load(fname + '.npz', allow_pickle=True) as f:
        xprotos, yprotos = f['xprotos'], f['yprotos']
        lamda = f['lamda']
        print(f"train accuracy: {f['accuracy_of_train_set'][-1]}, "
              f"\t validation accuracy: {f['accuracy_of_validation_set'][-1]} ({np.max(f['accuracy_of_validation_set'])})")

    return xprotos, yprotos, lamda

def get_class_weight(labels):
    return labels.shape[0] / (np.unique(labels).shape[0] * np.bincount(labels.astype('int64')))

if __name__ == '__main__':

    x = np.random.randn(4, 2)
    xx = np.repeat(np.expand_dims(x, axis=0), 3, axis=0)
    # print(x.shape, xx.shape, np.transpose(xx, (0, 2, 1)).shape)
    xxx = np.expand_dims(np.transpose(np.expand_dims(x, axis=0), (0, 2, 1)), axis=0) @ xx
    print(x.shape, xx.shape, xxx.shape)
    U, S, Vh = np.linalg.svd(x.T @ x, full_matrices=False, compute_uv=True, hermitian=False)
    U2, S2, Vh2 = np.linalg.svd(xxx, full_matrices=False, compute_uv=True, hermitian=False)

    # print('x x', U)
    # print('xxx', U2)
    # print('x x', Vh)
    # print('xxx', Vh2)
    print('x x', S)
    print('xxx', S2)
    print(U.shape, U2.shape)
    print(S2.shape)
    print(np.squeeze(S2).shape)
    y = np.random.randn(1,1,2,2,1)
    print(np.squeeze(y, axis=2).shape)
    # print(U.ndim, U2.ndim)
    # print(np.expand_dims(x, axis=(0,1)).shape)


