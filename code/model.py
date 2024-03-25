import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
from utils_model import *
from sklearn.metrics import confusion_matrix, accuracy_score# f1_score

def orthoganization_of_data(data_3d):
    for i, data in enumerate(data_3d):
        data_3d[i] = LA.orth(data)
    return data_3d


class Model():
    def __init__(self, dim_of_data, dim_of_subspace, **kwargs):
        if 'maxepochs' in kwargs.keys():
            self.nepochs = kwargs['maxepochs']
        else:
            self.nepochs = 100
        if 'metric_type' in kwargs.keys():
            self.metric_type = kwargs['metric_type']
        else:
            self.metric_type = 'geodesic'

        if 'nprotos' in kwargs.keys():
            self.nprotos = kwargs['nprotos']
        else:
            self.nprotos = 1
        if 'sigma' in kwargs.keys():
            self.sigma = kwargs['sigma']
        else:
            self.sigma = 1
        if 'actfun' in kwargs.keys():
            self.act_fun = kwargs['actfun']
        else:
            self.act_fun = 'sigmoid'
        if 'num_of_classes' in kwargs.keys():
            self.number_of_class = kwargs['num_of_classes']
        else:
            self.number_of_class = 2
        if 'balanced' not in kwargs.keys():
            self.balanced = None
        else:
            self.balanced = kwargs['balanced']
        if 'localized' in kwargs.keys():
            self.localized = kwargs['localized']
        else:
            self.localized = False

        self.dim_of_data = dim_of_data
        self.dim_of_subspace = dim_of_subspace

        self.low_bound_lambda = None
        self.xprotos_init = None
        self.lr_w, self.lr_r = None, None
        self.class_weights = None
        self.init_type = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-self.sigma * x))

    def loss(self, d_plus, d_minus):
        """
        Calculate the loss value for a signal data point.
        """
        if self.act_fun == 'identity':
            return (d_plus - d_minus) / (d_plus + d_minus)
        else:
            return self.sigmoid((d_plus - d_minus) / (d_plus + d_minus))

    def initialize_parameters(self, **kwargs):
        if ('xtrain' in kwargs.keys()) and ('ytrain' in kwargs.keys()):
            self.xprotos, self.yprotos = init_with_samples(
                kwargs['xtrain'],
                kwargs['ytrain'],
                self.nprotos
            )
            self.init_type = 'samples'
        else:
            assert ('classes' in kwargs.keys()), f"We need D, d and num. of classes."
            self.xprotos, self.yprotos = init_randn(
                self.dim_of_data,
                self.dim_of_subspace,
                kwargs['classes'],
                self.nprotos
            )
            self.init_type = 'random'

        self.xprotos_init = np.copy(self.xprotos)
        if self.localized:
            self.lamda = np.ones((self.xprotos.shape[0], self.xprotos.shape[-1])) / self.xprotos.shape[-1]
        else:
            self.lamda = np.ones((1, self.xprotos.shape[-1])) / self.xprotos.shape[-1]

    def get_distances_to_prototypes(self, xdata):
        """
        Compute the distances between input data and prototypes using the chordal distance based on canonical correlation.
        """
        return compute_distances_on_grassmann_mdf(xdata, self.xprotos, self.metric_type, relevance=self.lamda)

    def findWinner(self, data, label):
        """
        Find the closest prototypes to a given data point with the same/different labels.
        """

        results = self.get_distances_to_prototypes(data)

        distances = results['distance']
        sameclass = np.argwhere(self.yprotos == label).T[0]
        diffclass = np.argwhere(self.yprotos != label).T[0]

        iplus = sameclass[np.argmin(distances[sameclass])]
        iminus = diffclass[np.argmin(distances[diffclass])]
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

    def prediction(self, data):
        results = self.get_distances_to_prototypes(data)
        if data.ndim == 2:
            distances = np.expand_dims(results['distance'], axis=0)
        else:
            distances = results['distance']

        pred = np.zeros(data.shape[0])
        #NEW
        if distances.ndim == 1:
            iwinner = np.argmin(distances)
            pred[0] = self.yprotos[iwinner]
        else:
            for i, distance in enumerate(distances):
                iwinner = np.argmin(distance)
                pred[i] = self.yprotos[iwinner]
        return pred

    def metrics(self, labels, pred):
        assert labels.shape == pred.shape, f'their shape is labels: {labels.shape}, pred:{pred.shape}'

        acc = accuracy_score(labels, pred)
        conf_mat = confusion_matrix(
            labels, pred,
            normalize='true'
        )
        conf_mat = 100 * conf_mat if conf_mat[0, 0] <= 1 else conf_mat
        return 100 * acc, conf_mat

    def get_loss(self, X, Y):
        cost = 0
        for data, label in zip(X, Y):
            plus, minus = self.findWinner(data, label)
            cost += self.loss(plus['distance'], minus['distance'])
        return cost

    # ***********************************
    # ***** computing derivatives *******
    # ***********************************
    def der_act_fun(self, cost):
        """
        Compute the derivative of the activation function with respect to cost.
        """
        if self.act_fun == 'identity':
            return 1
        else:
            return self.sigma * cost * (1 - cost)

    def dE_distance_plus(self, cost, dplus, dminus):
        """
        Compute the derivative of the error function with respect to the distance to W^+ (winner prototype with the same label).
        """
        return 2 * self.der_act_fun(cost) * dminus / ((dplus + dminus) ** 2)

    def dE_distance_minus(self, cost, dplus, dminus):
        """
        Compute the derivative of the error function with respect to the distance to W^- (winner prototype with a different label).
        """
        return -2 * self.der_act_fun(cost) * dplus / ((dplus + dminus) ** 2)

    def der_W_chordal(self, X_rotated, canonicalcorrelation):
        """
        Compute the derivative of the distance (sum_i r_i * sin^2(theta_i)) with respect to W (the winner prototype).
        """
        Lam = np.tile(
            self.lamda[0] * canonicalcorrelation,
            (self.dim_of_data, 1)
        )
        return - 2 * Lam * X_rotated

    def der_W_pseudo_chordal(self, X_rotated): #CHECK !!!!!!
        """
        Compute the derivative of the distance (sum_i r_i * sin^2(theta_i)) with respect to W (the winner prototype).
        """
        Lam = np.tile(
            self.lamda[0],
            (self.dim_of_data, 1)
        )
        return - Lam * X_rotated

    def der_W_geodesic(self, X_rotated, canonicalcorrelation):
        """
        Compute the derivative of the distance (sum_i r_i * theta_i^2) with respect to W (the winner prototype).
        """
        G = 2 * np.diag(
            self.lamda[0] * np.arccos(canonicalcorrelation) / np.sqrt(1 - canonicalcorrelation ** 2)
        )
        return -X_rotated @ G

    def Euclidean_gradient(self, dE_dist, X_rot, CC):
        """
        Compute the (Euclidean) derivative of the error function with respect to the winner prototype.
        Args:
            dE_dist (float): Derivative of the error function with respect to the distance to the winner prototype.
        """
        if self.metric_type == 'chordal':
            return dE_dist * self.der_W_chordal(X_rot, CC)
        elif self.metric_type == 'pseudo-chordal':
            return dE_dist * self.der_W_pseudo_chordal(X_rot)
        else:
            return dE_dist * self.der_W_geodesic(X_rot, CC)

    def der_distance_relevance(self, canonicalcorrelation):
        """
        Compute the (Euclidean) derivative of the distance with respect to the relevance factors.
        Args:
            dE_dist (float): Derivative of the error function with respect to the distance to the winner prototype.
        """
        if self.metric_type == 'chordal':
            return - (canonicalcorrelation.T ** 2)
        elif self.metric_type == 'pseudo-chordal':
            return - canonicalcorrelation.T
        else:
            return np.arccos(canonicalcorrelation).T ** 2

    def fit(self, xtrain, ytrain, ** kwargs):
        """
        Perform one epoch of training using the provided training data.
        """

        if 'lr_w' in kwargs.keys():
            self.lr_w = kwargs['lr_w']
        else:
            self.lr_w = 0.01
        if 'lr_r' in kwargs.keys():
            self.lr_r = kwargs['lr_r']
        else:
            self.lr_r = self.lr_w / 100

        if 'low_bound_lambda' in kwargs.keys():
            self.low_bound_lambda = kwargs['low_bound_lambda']
        else:
            self.low_bound_lambda = 0.001

        if self.class_weights is None:
            if self.balanced:
                self.class_weights = np.power(get_class_weight(ytrain), 1)
                self.class_weights = self.class_weights / self.class_weights.sum()
            else:
                self.class_weights = np.ones(self.number_of_class)
            print("\nweights", self.class_weights, "\n")

        perm = np.random.permutation(xtrain.shape[0])
        for data, label in zip(xtrain[perm], ytrain[perm]):
            plus, minus = self.findWinner(data, label)
            cost = self.loss(plus['distance'], minus['distance'])

            # rotation of the coordinate system
            X_rot_plus = data @ plus['Q']
            X_rot_minus = data @ minus['Q']
            proto_rot_plus = self.xprotos[plus['index']] @ plus['Qw']
            proto_rot_minus = self.xprotos[minus['index']] @ minus['Qw']

            # Compute gradients
            dE_dist_plus = self.dE_distance_plus(cost, plus['distance'], minus['distance'])
            dE_dist_minus = self.dE_distance_minus(cost, plus['distance'], minus['distance'])

            Eucl_grad_plus = self.Euclidean_gradient(dE_dist_plus, X_rot_plus, plus['canonicalcorr'])
            Eucl_grad_minus = self.Euclidean_gradient(dE_dist_minus, X_rot_minus, minus['canonicalcorr'])

            # Update prototypes
            self.xprotos[plus['index']] = proto_rot_plus - self.class_weights[int(label)] * self.lr_w * Eucl_grad_plus
            self.xprotos[minus['index']] = proto_rot_minus - self.class_weights[int(label)] * self.lr_w * Eucl_grad_minus

            # Orthonormalization of prototypes
            self.xprotos[plus['index']] = LA.orth(self.xprotos[plus['index']])
            self.xprotos[minus['index']] = LA.orth(self.xprotos[minus['index']])

            # Update relevance factors
            if not self.localized:
                self.lamda[0] -= (
                    self.class_weights[int(label)] * self.lr_r * (
                        dE_dist_plus * self.der_distance_relevance(plus['canonicalcorr']) +
                        dE_dist_minus * self.der_distance_relevance(minus['canonicalcorr'])
                    )
                )
                # Normalization of relevance factors
                self.lamda[0, np.argwhere(self.lamda < self.low_bound_lambda)[:, 1]] = self.low_bound_lambda
                self.lamda[0] = self.lamda[0] / np.sum(self.lamda)
            else:
                self.lamda[plus['index']] -= (
                    self.class_weights[int(label)] * self.lr_r * (
                        dE_dist_plus * self.der_distance_relevance(plus['canonicalcorr'])
                    )
                )
                self.lamda[minus['index']] -= (
                    self.lr_r * (
                        dE_dist_minus * self.der_distance_relevance(minus['canonicalcorr'])
                    )
                )
                self.lamda[plus['index'], np.argwhere(self.lamda[plus['index']] < self.low_bound_lambda).T[0]] = self.low_bound_lambda
                self.lamda[plus['index']] = self.lamda[plus['index']] / np.sum(self.lamda[plus['index']])
                self.lamda[minus['index'], np.argwhere(self.lamda[minus['index']] < self.low_bound_lambda).T[0]] = self.low_bound_lambda
                self.lamda[minus['index']] = self.lamda[minus['index']] / np.sum(self.lamda[minus['index']])




    def save_results(
            self, fname, acc_train, acc_val, conf_mat=None, cost_train=None, cost_val=None):
        with open(fname+'.npz', 'wb') as f:
            np.savez(
                f,
                xprotos_init=self.xprotos_init,
                xprotos=self.xprotos, yprotos=self.yprotos,
                lamda=self.lamda,
                number_of_epochs=self.nepochs, init_protos_method=self.init_type,
                learning_rate_w=self.lr_w, learning_rate_r=self.lr_r,
                number_of_prototypes=self.nprotos, sigma_sigmoid=self.sigma,
                number_of_class=self.number_of_class, dim_of_data=self.dim_of_data,
                dim_of_subspace=self.dim_of_subspace,
                accuracy_of_train_set=acc_train,
                accuracy_of_validation_set=acc_val,
                conf_mat=conf_mat,
                cost_of_train_set=cost_train, cost_of_validation_set=cost_val,
            )



def load_results(fname):
    with np.load(fname + '.npz') as file:
        xprotos = file['xprotos']
    # for i in range(xprotos.shape[0]):
    #     plotprototype(xprotos[i], fname, i)

# if __name__ == '__main__':
#     load_results('eth80')