import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import linalg as LA
import os
from preprocessing import *


class Model():
    def __init__(self, dim, dim_of_subspace, numclasses, **kwargs):
        if 'maxepochs' in kwargs.keys():
            self.nepochs = kwargs['maxepochs']
        else:
            self.nepochs = 100
        if 'distance' in kwargs.keys():
            self.distance = kwargs['distance']
        else:
            self.distance = 'chordal'
        if 'lr_w' in kwargs.keys():
            self.lr_w = kwargs['lr_w']
        else:
            self.lr_w = 0.01
        if 'lr_r' in kwargs.keys():
            self.lr_r = kwargs['lr_r']
        else:
            self.lr_r = self.lr_w / 100
        if 'decay_rate' in kwargs.keys():
            self.decay_rate = kwargs['decay_rate']
        else:
            self.decay_rate = 1
        if 'regularizer_coef' in kwargs.keys():
            self.regularizer_coef = kwargs['regularizer_coef']
        else:
            self.regularizer_coef = 0
        if 'initmethod' not in kwargs.keys():
            self.initmethod = 'pca'
        else:
            self.initmethod = kwargs['initmethod']
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
        if 'plot_res' in kwargs.keys():
            self.plot_res = kwargs['plot_res']
        else:
            self.plot_res = 10
        self.number_of_class = numclasses
        self.dim_of_data = dim
        self.dim_of_subspace = dim_of_subspace

        self.lamda = np.ones((1, self.dim_of_subspace)) / self.dim_of_subspace
        self.acc_train = np.zeros(self.nepochs+1)
        self.cost_train = np.zeros(self.nepochs+1)
        self.acc_val = np.zeros(self.nepochs+1)
        self.cost_val = np.zeros(self.nepochs+1)

    def init_protos_random(self, labels):
        """
        use gaussian distribution to initialize prototypes
        :return: initial value for prototypes
        """
        self.xprotos = np.random.normal(0, 1,
                                        (
                                            self.number_of_class * self.nprotos,
                                            self.dim_of_data,
                                            self.dim_of_subspace
                                        ))
        self.yprotos = np.zeros(self.number_of_class * self.nprotos)
        k = 0
        for label in np.unique(labels):
            for j in range(self.nprotos):
                self.xprotos[k] = LA.orth(self.xprotos[k])
                self.yprotos[k] = label
                k += 1

    def pca_model(self, data, d):
        """
        apply PCA on each class separately to initialize prototypes
        :param data: list of samples (d * n)
        :return: initial value for prototypes
        """
        pca = PCA(n_components=d)
        pca.fit(data.T)
        return LA.orth(pca.components_.T)

    def init_protos_pca(self, data, labels):
        """
        apply PCA on each class separately to initialize prototypes
        :param data: list of samples (n * D * d)
        :param labels
        :return: initial value for prototypes
        """

        self.xprotos = np.zeros((self.number_of_class * self.nprotos, self.dim_of_data,
                                 self.dim_of_subspace))
        self.yprotos = np.zeros(self.number_of_class * self.nprotos)
        t = 0
        for (i, label) in enumerate(np.unique(labels)):
            idx = np.random.permutation(np.argwhere(labels == label).T[0])
            tmp = np.zeros((self.dim_of_data, self.dim_of_subspace * len(idx)))
            for j in range(len(idx)):
                tmp[:, j * self.dim_of_subspace:(j + 1) * self.dim_of_subspace] = data[idx[j]]

            self.yprotos[t: t + self.nprotos] = label
            n = len(idx) // self.nprotos
            for k in range(self.nprotos):
                self.xprotos[t] = self.pca_model(tmp[:, k * self.dim_of_subspace *n: (k+1)* self.dim_of_subspace*n],
                                                 self.dim_of_subspace)
                t = t + 1


    def init_protos_samples(self, data, labels):
        """
        initialize prototypes with data points
        :param data: list of samples (n * D * d)
        :param labels
        :return: initial value for prototypes
        """
        eps=0.0001

        self.xprotos = np.zeros((self.number_of_class * self.nprotos,
                                self.dim_of_data, self.dim_of_subspace))
        self.yprotos = np.zeros(self.number_of_class * self.nprotos)
        t = 0
        for (i, label) in enumerate(np.unique(labels)):
            idx = np.argwhere(labels == label).T[0]
            tmp = data[np.random.permutation(idx)[:self.nprotos], :, :] + eps * \
                  np.random.randn(self.nprotos, self.dim_of_data, self.dim_of_subspace)
            self.yprotos[t: t + self.nprotos] = label
            for k in range(self.nprotos):
                self.xprotos[t] = LA.orth(tmp[k])
                t = t+1


    def init_protos_kmeans(self, data, labels):
        """
        initialize prototypes with data points
        :param data: list of samples (n * D * d)
        :param labels
        :return: initial value for prototypes
        """
        model_kmeans = \
            kmeans(
                nprotos=self.nprotos,
                distancetype='geodesic',
                dim_of_data=data.shape[1],  # the dimensionality of data
                dim_of_subspace=data.shape[2],
                lr=0.005,  # learning rate for prototypes
                init_method='random',  # 'random' or 'samples' or 'principal'<- does not work well (for eth80)
                nepochs=100,  # maximum number of epochs
                print_res=1000,
            )

        self.xprotos = np.zeros((self.number_of_class * self.nprotos,
                                self.dim_of_data, self.dim_of_subspace))
        self.yprotos = np.zeros(self.number_of_class * self.nprotos)
        t = 0
        for (i, label) in enumerate(np.unique(labels)):
            idx = np.argwhere(labels == label).T[0]
            model_kmeans.fit(
                data[idx]
            )
            self.yprotos[t: t + self.nprotos] = label
            for k in range(self.nprotos):
                self.xprotos[t] = LA.orth(model_kmeans.xprotos[k])
                t = t+1


    def init_protos_principal_subspace(self, data, labels):
        """
        initialize prototypes with data points
        :param data: list of samples (n * D * d)
        :param labels
        :return: initial value for prototypes
        """

        assert self.nprotos==1
        self.xprotos = np.zeros((self.number_of_class * self.nprotos,
                                self.dim_of_data, self.dim_of_subspace))
        self.yprotos = np.zeros(self.number_of_class * self.nprotos)
        for (i, label) in enumerate(np.unique(labels)):
            idx = np.argwhere(labels == label).T[0]
            tmp = data[idx]
            self.yprotos[i] = label
            tmp2 = np.zeros((self.dim_of_data, self.dim_of_data))
            for k in range(len(tmp)):
                tmp2 = tmp2 + np.matmul(tmp[k], tmp[k].T)
            S, U = LA.eig(tmp2)
            ids = np.argsort(S)
            self.xprotos[i] = U[:, ids[-1:-self.dim_of_subspace-1:-1]]



    # def adaptive_learning_rate_drop(self, lr_init, epoch, n=50):
    #     return lr_init * (self.decay_rate ** np.floor(epoch / n))
    #
    # def adaptive_learning_rate_exp(self, lr_init, epoch):
    #     return lr_init * np.exp(- self.decay_rate * epoch)

    def sigmoid(self, x):
        """
        sigmoid function (used in the cost function)
        :param x: independent varaible
        :return: sigmoid function value 1/(1+exp(-sigma*x))
        """
        return 1 / (1 + np.exp(-self.sigma * x))

    def loss(self, d_plus, d_minus):
        """
        the cost value for a signal data point
        :param d_plus:
        :param d_minus:
        :return:
        """
        if self.act_fun == 'identity':
            return (d_plus - d_minus) / (d_plus + d_minus) - \
                   self.regularizer_coef * np.sum(np.log(self.lamda))
        else:
            return self.sigmoid((d_plus - d_minus) / (d_plus + d_minus)) - \
                   self.regularizer_coef * np.sum(np.log(self.lamda))

    def der_act_fun(self, cost):
        """
        it computes the derivative of activation function: df(x) / dx
        :param cost:
        :return:
        """
        if self.act_fun == 'identity':
            return 1
        else:
            return self.sigma * cost * (1 - cost)

    def chordal_distance(self, orth1):
        """
        it computes the chordal distance between two subspaces (using canonical correlation)
        :param orth1: the first subspace, a 2d array of size (d * d)
        :param type: chordal or pseudo chordal
        :return: canonical correlation, chordal distance and principal directions
        """
        output = dict()
        output['principal_1'] = np.zeros((self.xprotos.shape[0], self.dim_of_subspace,
                                          self.dim_of_subspace))
        output['principal_2'] = np.zeros(output['principal_1'].shape)
        output['distance'] = np.zeros((self.xprotos.shape[0], 1))
        output['canonicalcorr'] = np.zeros((self.xprotos.shape[0], self.dim_of_subspace))

        for i in range(self.yprotos.shape[0]):
            m = np.matmul(orth1.T, self.xprotos[i])
            U, S, Vh = LA.svd(m, full_matrices=False, compute_uv=True, overwrite_a=False,
                              lapack_driver='gesdd')  # MATLAB and Octave use the 'gesvd' approach
            output['principal_1'][i] = U; output['principal_2'][i] = Vh.T
            if self.lamda.shape[1] == 1:
                if self.distance=='pseudo chordal':
                    output['distance'][i] = 1 - S[0]
                else:
                    output['distance'][i] = 1 - S[0] ** 2
            else:
                output['canonicalcorr'][i] = S
                if self.distance=='pseudo chordal':
                    output['distance'][i] = np.sum(self.lamda) - \
                                            np.matmul(self.lamda, np.expand_dims(S, axis=1))
                else:
                    output['distance'][i] = np.sum(self.lamda) - \
                                            np.matmul(self.lamda, np.expand_dims(S ** 2, axis=1))
            assert output['distance'][i] >= 0, "distance is negative"
        return output

    def geodesic_distance(self, orth1):
        """
        it computes the chordal distance between two subspaces (using canonical correlation)
        :param orth1: the first subspace, a 2d array of size (d * d)
        :return: canonical correlation, chordal distance and principal directions
        """
        output = dict()
        output['principal_1'] = np.zeros((self.xprotos.shape[0], self.dim_of_subspace,
                                          self.dim_of_subspace))
        output['principal_2'] = np.zeros(output['principal_1'].shape)
        output['distance'] = np.zeros((self.xprotos.shape[0], 1))
        output['canonicalcorr'] = np.zeros((self.xprotos.shape[0], self.dim_of_subspace))

        for i in range(self.yprotos.shape[0]):
            m = np.matmul(orth1.T, self.xprotos[i])
            U, S, Vh = LA.svd(m, full_matrices=False, compute_uv=True, overwrite_a=False,
                              lapack_driver='gesdd')
            output['principal_1'][i] = U; output['principal_2'][i] = Vh.T
            if self.lamda.shape[1] == 1:
                if self.distance=='pseudo geodesic':
                    output['distance'][i] = np.arccos(S[0])
                else:
                    output['distance'][i] = np.arccos(S[0]) ** 2
            else:
                output['canonicalcorr'][i] = S
                S = np.array([s if s <= 1 else 1 for s in S])  # CHECK
                if self.distance=='pseudo geodesic':
                    output['distance'][i] = np.matmul(
                                                      self.lamda,
                                                      np.expand_dims(
                                                          np.arccos(S), axis=1,
                                                      ))
                else:
                    output['distance'][i] = np.matmul(
                        self.lamda,
                        np.expand_dims(
                            np.arccos(S) ** 2, axis=1,
                        ))

                    # if np.isnan(output['distance'][i]):
                    #     output['distance'][i] = 0

            assert output['distance'][i] >= 0, "distance is negative"
        return output

    def findWinner(self, datapoint, label):
        """
        It find the closest prototypes to datapoint with the same/different labels
        :param datapoint: a subspace, a 2d array (d * d)
        :param label: the label of the data point, an integer number
        :return: the closest protos with the same/different labels (plus, minus)
        """
        if self.distance == 'geodesic':
            results = self.geodesic_distance(datapoint)
        elif self.distance == 'pseudo geodesic':
            results = self.geodesic_distance(datapoint)
        elif self.distance == 'pseudo chordal':
            results = self.chordal_distance(datapoint)
        else:
            results = self.chordal_distance(datapoint)

        distances = results['distance']
        sameclass = np.argwhere(self.yprotos == label).T[0]
        diffclass = np.argwhere(self.yprotos != label).T[0]

        iplus = sameclass[np.argmin(distances[sameclass])]
        dplus = distances[iplus]
        iminus = diffclass[np.argmin(distances[diffclass])]
        dminus = distances[iminus]

        plus = dict()
        plus['index'] = iplus
        plus['distance'] = dplus
        plus['Q'] = results['principal_1'][iplus]
        plus['Qw'] = results['principal_2'][iplus]
        plus['canonicalcorr'] = results['canonicalcorr'][iplus]

        minus = dict()
        minus['index'] = iminus
        minus['distance'] = dminus
        minus['Q'] = results['principal_1'][iminus]
        minus['Qw'] = results['principal_2'][iminus]
        minus['canonicalcorr'] = results['canonicalcorr'][iminus]
        return plus, minus

    def evaluate(self, data, labels):
        acc = 0
        cost = 0

        # check whether the data points are single image or a set of images
        if len(data.shape)==2:
            data = np.expand_dims(data, axis=2)
            for i in range(len(labels)):
                s = np.zeros(self.xprotos.shape[0])
                data[i] = data[i]/ LA.norm(data[i])
                for j in range(len(s)):
                    U, S, Vh = LA.svd(np.matmul(data[i].T, self.xprotos[j]), full_matrices=False, compute_uv=True, overwrite_a=False,
                                      lapack_driver='gesdd')
                    s[j] = S[0]
                m = np.argmax(s)
                if self.yprotos[m]==labels[i]:
                    acc +=1
        else:
            for i in range(len(labels)):
                plus, minus = self.findWinner(data[i], labels[i])
                cost += self.loss(plus['distance'], minus['distance'])[0]
                if plus['distance'] < minus['distance']:
                    acc += 1
        acc /= (len(labels) / 100)
        cost /= len(labels)

        return acc, cost

    def orthoganization_of_data(self, xtrain):
        for i in range(xtrain.shape[0]):
            xtrain[i] = LA.orth(xtrain[i])
        return xtrain

    def dE_distance_plus(self, cost, dplus, dminus):
        return 2 * self.der_act_fun(cost) * dminus / ((dplus + dminus) ** 2)

    def dE_distance_minus(self, cost, dplus, dminus):
        """
        computes the derivative of the cost function w.r.t distance to the prototype with different label
        :param cost: cost value
        :param dplus: distance to the closest prototype with the same label
        :param dminus: distance to the closest prototype with different label
        :return: dE / dD^-
        """
        return - 2 * self.der_act_fun(cost) * dplus / ((dplus + dminus) ** 2)

    def der_W_pseudo_chordal(self, X_rotated):
        """
        derivative of distance (1 - sum_i l_i * cos(t_i))
            w.r.t the prototype (principal component of the prototype)
        :param X_rotated:  principal direction
        :return: d D / dW
        """
        Lam = np.tile(self.lamda[0],
                        (self.dim_of_data, 1))
        return - Lam * X_rotated

    def der_W_chordal(self, X_rotated, canonicalcorr):
        """
        derivative of distance (sum_i l_i * sin^2(t_i))
            w.r.t the prototype (principal component of the prototype)
        :param X_rotated:  principal direction
        :return: d D / dW
        """
        Lam = np.tile(self.lamda[0] * canonicalcorr,
                        (self.dim_of_data, 1))
        return - 2 * Lam * X_rotated

    def der_W_pseudo_geodesic(self, X_rotated, canonicalcorr):
        """
        derivative of distance (sum_i l_i * t_i)
            w.r.t the prototype (principal component of the prototype)
        :param X_rotated: principal direction
        :param canonicalcorr: canonical correlation (i.e. cos(t))
        :return: d D / dW
        """
        # Lam = np.tile(self.lamda[0], (self.dim_of_data, 1))
        # den = np.tile(np.sqrt(1 - canonicalcorr ** 2), (self.dim_of_data, 1))
        # return - Lam * X_rotated / den
        G = np.diag(self.lamda[0] / np.sqrt(1 - canonicalcorr ** 2))
        return - np.matmul(X_rotated, G)

    def der_W_geodesic(self, X_rotated, canonicalcorr):
        """
        derivative of distance (sum_i l_i * t_i^2)
            w.r.t the prototype (principal component of the prototype)
        :param X_rotated: principal direction
        :param canonicalcorr: canonical correlation (i.e. cos(t))
        :return: d D / dW
        """

        G = 2 * np.diag(self.lamda[0] * np.arccos(canonicalcorr) / np.sqrt(1 - canonicalcorr ** 2))
        return - np.matmul(X_rotated, G)

    def Euclidean_gradient_geodesic(self, dE_dist, X_rot, CC):
        return dE_dist * self.der_W_geodesic(X_rot, CC)

    def Riemannian_gradient_geodesic(self, Eucl_grad_W, W):
        return np.matmul(np.eye(W.shape[0]) - np.matmul(W, W.T) , Eucl_grad_W)

    def Update_Grassman(self, W_old, grad, sign=1):
        U, S, Vh = LA.svd(grad, full_matrices=False, compute_uv=True, overwrite_a=False, lapack_driver='gesdd')
        return (W_old @ Vh @ np.diag(np.cos(sign*self.lr_w * S))  + U @ np.diag(np.sin(sign * self.lr_w * S)) ) @ Vh.T



    def der_lamda_pseudo_chordal(self, canonicalcorr):
        """
        derivative of chordal distance
            w.r.t the lambda (direction importance)
        :param canonicalcorr: principal direction
        :return: d D / dl
        """
        return - canonicalcorr.T

    def der_lamda_chordal(self, canonicalcorr):
        return - (canonicalcorr.T ** 2)

    def der_lamda_pseudo_geodesic(self, canonicalcorr):
        return np.arccos(canonicalcorr).T

    def der_lamda_geodesic(self, canonicalcorr):
        return np.arccos(canonicalcorr).T ** 2

    def der_lamda_regularization_term(self):
        if self.regularizer_coef == 0:
            return 0
        else:
            return - self.regularizer_coef / self.lamda[0]

    def one_epoch(self, xtrain, ytrain, epoch, low_bound_lambda=0.001):
        lr_w = self.lr_w #self.adaptive_learning_rate_drop(self.lr_w, epoch, n=20)
        lr_r = self.lr_r #self.adaptive_learning_rate_drop(self.lr_r, epoch, n=20)

        for i in np.random.permutation(xtrain.shape[0]):
            plus, minus = self.findWinner(xtrain[i], ytrain[i])
            cost = self.loss(plus['distance'], minus['distance'])[0]

            # rotation of the coordinate system
            X_rot_plus = np.matmul(xtrain[i], plus['Q'])
            X_rot_minus = np.matmul(xtrain[i], minus['Q'])
            proto_rot_plus = np.matmul(self.xprotos[plus['index']], plus['Qw'])
            proto_rot_minus = np.matmul(self.xprotos[minus['index']], minus['Qw'])

            # common details in updating rules
            dE_dist_plus = self.dE_distance_plus(cost, plus['distance'][0], minus['distance'][0])
            dE_dist_minus = self.dE_distance_minus(cost, plus['distance'][0], minus['distance'][0])

            # updating prototypes
            if self.distance == 'geodesic':
                Eucl_grad_plus = self.Euclidean_gradient_geodesic(dE_dist_plus, X_rot_plus, plus['canonicalcorr'])
                    # self.der_W_geodesic(X_rot_plus, plus['canonicalcorr'])
                Eucl_grad_minus = self.Euclidean_gradient_geodesic(dE_dist_minus, X_rot_minus, minus['canonicalcorr'])
                    # self.der_W_geodesic(X_rot_minus, minus['canonicalcorr'])
                self.xprotos[plus['index']] = proto_rot_plus - lr_w * \
                                              Eucl_grad_plus
                                              # Eucl_grad_plus
                                              # self.Riemannian_gradient_geodesic(Eucl_grad_plus, proto_rot_plus)

                self.xprotos[minus['index']] = proto_rot_minus - lr_w * \
                                               Eucl_grad_minus
                                               # Eucl_grad_minus
                                               # self.Riemannian_gradient_geodesic(Eucl_grad_minus, proto_rot_minus)

                # self.xprotos[plus['index']] = self.Update_Grassman(self.xprotos[plus['index']],
                #                                                    self.Riemannian_gradient_geodesic(Eucl_grad_plus,
                #                                                                                      proto_rot_plus),
                #                                                    1)
                #
                # self.xprotos[minus['index']] = self.Update_Grassman(self.xprotos[minus['index']],
                #                                                     self.Riemannian_gradient_geodesic(Eucl_grad_minus,
                #                                                                                      proto_rot_minus),
                #                                                     1)

                self.lamda[0] = self.lamda[0] - lr_r * (
                        dE_dist_plus * self.der_lamda_geodesic(plus['canonicalcorr']) + \
                        dE_dist_minus * self.der_lamda_geodesic(minus['canonicalcorr']) +
                        self.der_lamda_regularization_term())
            elif self.distance == 'pseudo geodesic':
                self.xprotos[plus['index']] = proto_rot_plus - lr_w * dE_dist_plus * \
                                              self.der_W_pseudo_geodesic(X_rot_plus)
                self.xprotos[minus['index']] = proto_rot_minus - lr_w * dE_dist_minus * \
                                               self.der_W_pseudo_geodesic(X_rot_minus)

                self.lamda[0] = self.lamda[0] - lr_r * (
                            dE_dist_plus * self.der_lamda_pseudo_geodesic(plus['canonicalcorr']) + \
                            dE_dist_minus * self.der_lamda_pseudo_geodesic(minus['canonicalcorr']))
            elif self.distance == 'pseudo chordal':
                self.xprotos[plus['index']] = proto_rot_plus - lr_w * dE_dist_plus * \
                                              self.der_W_pseudo_chordal(X_rot_plus)
                self.xprotos[minus['index']] = proto_rot_minus - lr_w * dE_dist_minus * \
                                               self.der_W_pseudo_chordal(X_rot_minus)
                self.lamda[0] = self.lamda[0] - lr_r * (
                            dE_dist_plus * self.der_lamda_pseudo_chordal(plus['canonicalcorr']) + \
                            dE_dist_minus * self.der_lamda_pseudo_chordal(minus['canonicalcorr']))
            else:
                self.xprotos[plus['index']] = proto_rot_plus - lr_w * dE_dist_plus * \
                                              self.der_W_chordal(X_rot_plus, plus['canonicalcorr'])
                self.xprotos[minus['index']] = proto_rot_minus - lr_w * dE_dist_minus * \
                                               self.der_W_chordal(X_rot_minus, minus['canonicalcorr'])
                self.lamda[0] = self.lamda[0] - lr_r * (
                        dE_dist_plus * self.der_lamda_chordal(plus['canonicalcorr']) + \
                        dE_dist_minus * self.der_lamda_chordal(minus['canonicalcorr']))



            # orthonormalization prototypes
            self.xprotos[plus['index']] = LA.orth(self.xprotos[plus['index']])
            self.xprotos[minus['index']] = LA.orth(self.xprotos[minus['index']])

            # normalization of relevance vector: lamda
            self.lamda[0, np.argwhere(self.lamda < low_bound_lambda)[:,1]] = low_bound_lambda
            self.lamda[0] = self.lamda[0] / np.sum(self.lamda)


    def fit(self, xtrain, ytrain, **kwargs):
        if 'xval' in kwargs.keys():
            xval = kwargs['xval']
            yval = kwargs['yval']
        else:
            xval = np.array([])
            yval = np.array([])
            self.acc_val = np.array([])
            self.cost_val = np.array([])
        if 'isRGB' in kwargs.keys():
            self.isRGB = kwargs['isRGB']
        else:
            self.isRGB = True

        # ************** Initializing prototypes
        if (self.initmethod == 'pca'):
            self.init_protos_pca(xtrain, ytrain)
        elif (self.initmethod == 'samples'):
            self.init_protos_samples(xtrain, ytrain)
        elif (self.initmethod == 'principal'):
            self.init_protos_principal_subspace(xtrain, ytrain)
        elif (self.initmethod == 'kmeans'):
            self.init_protos_kmeans(xtrain, ytrain)
        else:
            self.init_protos_random(ytrain)


        self.acc_train[0], self.cost_train[0] = self.evaluate(xtrain, ytrain)
        if xval.size != 0:
            self.acc_val[0], self.cost_val[0] = self.evaluate(xval, yval)

        # ************** training
        for epoch in range(1, self.nepochs + 1):
            self.one_epoch(xtrain, ytrain, epoch)
            self.acc_train[epoch], self.cost_train[epoch] = self.evaluate(xtrain, ytrain)
            if xval.size != 0:
                self.acc_val[epoch], self.cost_val[epoch] = self.evaluate(xval, yval)
            np.set_printoptions(precision=3)
            if epoch % 20 == 0:
                print("relevance vector: ", self.lamda[0])
                if xval.size != 0:
                    print("epoch {}: \t training accuracy: {:.2f}, \t testing accuracy: {:.2f} (max: {:.5f})".format(
                        epoch, self.acc_train[epoch], self.acc_val[epoch], np.max(self.acc_val[:epoch+1])))
                else:
                    print("epoch {}: \t accuracy: {:.2f}, \t cost: {:.2f}".format(
                        epoch, self.acc_train[epoch], self.cost_train[epoch]))

            # ************** plot error curve
            if epoch % self.plot_res == 0:
                # self.errorcurves(epoch)
                # self.plotprototype()
                if 'fname' in kwargs.keys():
                    self.save_results(kwargs['fname'])

            # if ~np.all(self.lamda):
            #     break

    def errorcurves(self, epoch):
        nepochs = list(range(epoch+1))
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(nepochs, self.acc_train[nepochs], label='train set')
        if self.acc_val.size != 0:
            ax[0].plot(nepochs, self.acc_val[nepochs], label='validation set')
        ax[0].set_title('accuracy')
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel('accuracy')
        ax[0].tick_params(bottom=True, top=False, left=True, right=True)

        ax[1].plot(nepochs, self.cost_train[nepochs], label='train set')
        if self.acc_val.size != 0:
            ax[1].plot(nepochs, self.cost_val[nepochs], label='validation set')
        ax[1].set_title('cost function')
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel('cost value')
        plt.legend()

        ax[2].plot(list(range(self.lamda.shape[1])),
                   self.lamda[0], label='dir. importance')
        ax[2].set_title('lambda')
        ax[2].set_xlabel('index of dim')
        ax[2].set_ylabel('importance')
        plt.legend()

        plt.show()

    def plotprototype(self):
        c = np.random.randint(len(np.unique(self.yprotos)), size=1)[0]

        ncols = 5
        nrows = self.xprotos[c].shape[1] // ncols
        fig, ax = plt.subplots(nrows, ncols, figsize=(15, 10))
        k = 0
        for i in range(nrows):
            for j in range(ncols):
                if self.isRGB:
                    m = np.min(self.xprotos[c, :, k])
                    M = np.max(self.xprotos[c, :, k])
                    pr = (self.xprotos[c, :, k] - m) / (M-m)
                    if nrows==1:
                        ax[j].imshow(
                            Vec2FIG(pr, isRGB=True),
                            vmin=self.xprotos[c, :, k].min(),
                            vmax=self.xprotos[c, :, k].max()
                        )
                    else:
                        ax[i, j].imshow(
                            Vec2FIG(pr, isRGB=True),
                            vmin=self.xprotos[c, :, k].min(),
                            vmax=self.xprotos[c, :, k].max()
                        )
                else:
                    if nrows==1:
                        ax[j].imshow(
                            Vec2FIG(self.xprotos[c, :, k], isRGB=False),
                            cmap=plt.get_cmap('gray'),
                            vmin=self.xprotos[c, :, k].min(),
                            vmax=self.xprotos[c, :, k].max()
                        )
                    else:
                        ax[i, j].imshow(
                            Vec2FIG(self.xprotos[c, :, k], isRGB=False),
                            cmap=plt.get_cmap('gray'),
                            vmin=self.xprotos[c, :, k].min(),
                            vmax=self.xprotos[c, :, k].max()
                        )
                k += 1
        plt.suptitle("class: {}".format(self.yprotos[c]), fontsize=30)
        plt.show()

    def save_results(self, fname):
        with open(fname+'.npz', 'wb') as f:
            np.savez(f,
                     xprotos=self.xprotos, yprotos=self.yprotos,
                     lamda=self.lamda,
                     number_of_epochs=self.nepochs, init_protos_method=self.initmethod,
                     learning_rate_w=self.lr_w, learning_rate_r=self.lr_r,
                     number_of_prototypes=self.nprotos, sigma_sigmoid=self.sigma,
                     number_of_class=self.number_of_class, dim_of_data=self.dim_of_data,
                     dim_of_subspace=self.dim_of_subspace,
                     accuracy_of_train_set=self.acc_train,
                     cost_of_train_set=self.cost_train,
                     accuracy_of_validation_set=self.acc_val,
                     cost_of_validation_set=self.cost_val
                     )

class kmeans(): #data, labels):
    def __init__(self, dim_of_data, dim_of_subspace, lr=0.001, **kwargs):
        if 'nprotos' in kwargs.keys():
            self.nprotos = kwargs['nprotos']
        else:
            self.nprotos = 1
        if 'distancetype' in kwargs.keys():
            self.distancetype = kwargs['distancetype']
        else:
            self.distancetype = 'geodesic'
        if 'nepochs' in kwargs.keys():
            self.nepochs = kwargs['nepochs']
        else:
            self.nepochs = 100
        if 'init_method' in kwargs.keys():
            self.initmethod = kwargs['init_method']
        else:
            self.initmethod = 'samples'
        if 'print_res' in kwargs.keys():
            self.print_res = kwargs['print_res']
        else:
            self.print_res = 1000

        self.dim_of_data = dim_of_data
        self.dim_of_subspace = dim_of_subspace
        self.lr = lr
        self.cost_train = np.zeros(self.nepochs + 1)

    def init_protos_samples(self, data):
        eps = 0.0001
        self.xprotos = data[np.random.permutation(data.shape[0])[:self.nprotos], :, :] + eps * \
                        np.random.randn(self.nprotos, self.dim_of_data, self.dim_of_subspace)

    def init_protos_random(self):
        """
        use gaussian distribution to initialize prototypes
        :return: initial value for prototypes
        """
        self.xprotos = np.random.normal(0, 1,
                                        (
                                            self.nprotos,
                                            self.dim_of_data,
                                            self.dim_of_subspace
                                        ))
        self.xprotos = self.orthoganization_of_data(self.xprotos)


    def orthoganization_of_data(self, xtrain):
        for i in range(xtrain.shape[0]):
            xtrain[i] = LA.orth(xtrain[i])
        return xtrain

    def geodesic_distance(self, orth1):
        """
        it computes the chordal distance between two subspaces (using canonical correlation)
        :param orth1: the first subspace, a 2d array of size (d * d)
        :return: canonical correlation, chordal distance and principal directions
        """
        output = dict()
        output['principal_1'] = np.zeros((self.nprotos, self.dim_of_subspace,
                                          self.dim_of_subspace))
        output['principal_2'] = np.zeros(output['principal_1'].shape)
        output['distance'] = np.zeros((self.nprotos, 1))
        output['canonicalcorr'] = np.zeros((self.nprotos, self.dim_of_subspace))

        for i in range(self.nprotos):
            # print(orth1.shape, self.xprotos[i].shape)
            m = np.matmul(orth1.T, self.xprotos[i])
            U, S, Vh = LA.svd(m, full_matrices=False, compute_uv=True, overwrite_a=False,
                              lapack_driver='gesdd')
            output['principal_1'][i] = U; output['principal_2'][i] = Vh.T
            if self.xprotos.shape[2] == 1:
                output['distance'][i] = np.arccos(S[0]) ** 2
            else:
                output['canonicalcorr'][i] = S
                S = np.array([s if s<=1 else 1 for s in S])
                # print(S, "cc", np.arccos(S))
                output['distance'][i] = np.sum(np.arccos(S) ** 2) #np.expand_dims(np.arccos(S) ** 2, axis=1)
            assert output['distance'][i] >= 0, "distance is negative"
        return output

    def findWinner(self, datapoint):
        """
        It find the closest prototypes to datapoint with the same/different labels
        :param datapoint: a subspace, a 2d array (d * d)
        :param label: the label of the data point, an integer number
        :return: the closest protos with the same/different labels (plus, minus)
        """
        if self.distancetype == 'geodesic':
            results = self.geodesic_distance(datapoint)
        # elif self.distance == 'pseudo chordal':
        #     results = self.chordal_distance(datapoint)
        else:
            results = self.geodesic_distance(datapoint)

        distances = results['distance']
        # sameclass = np.argwhere(self.yprotos == label).T[0]
        # diffclass = np.argwhere(self.yprotos != label).T[0]

        iplus = np.argmin(distances) #sameclass[np.argmin(distances)]
        dplus = distances[iplus]

        plus = dict()
        plus['index'] = iplus
        plus['distance'] = dplus
        plus['Q'] = results['principal_1'][iplus]
        plus['Qw'] = results['principal_2'][iplus]
        plus['canonicalcorr'] = results['canonicalcorr'][iplus]
        return plus

    def der_W_geodesic(self, X_rotated, canonicalcorr):
        """
        derivative of distance (sum_i l_i * t_i^2)
            w.r.t the prototype (principal component of the prototype)
        :param X_rotated: principal direction
        :param canonicalcorr: canonical correlation (i.e. cos(t))
        :return: d D / dW
        """
        G = 2 * np.diag(np.arccos(canonicalcorr) / np.sqrt(1 - canonicalcorr ** 2))
        return - np.matmul(X_rotated, G)

    def evaluate(self, data):
        cost = 0
        for i in range(data.shape[0]):
            winner = self.findWinner(data[i])
            cost += winner['distance']
        return cost

    def one_epoch(self, data):
        for i in np.random.permutation(data.shape[0]):
            # assignment step
            winner = self.findWinner(data[i])

            # rotation of the coordinate system
            X_rot_plus = np.matmul(data[i], winner['Q'])
            proto_rot_winner = np.matmul(self.xprotos[winner['index']], winner['Qw'])

            # update step
            # if self.distance == 'geodesic':
            self.xprotos[winner['index']] = proto_rot_winner - self.lr * \
                                            self.der_W_geodesic(X_rot_plus, winner['canonicalcorr'])
            # orthonormalization prototypes
            self.xprotos[winner['index']] = LA.orth(self.xprotos[winner['index']])

    def fit(self, data, **kwargs):
        # initializing prototypes
        if (self.initmethod == 'random'):
            self.init_protos_random()
        else:
            self.init_protos_samples(data)

        self.cost_train[0] = self.evaluate(data)
        for epoch in range(1, self.nepochs+1):
            self.one_epoch(data)
            self.cost_train[epoch] = self.evaluate(data)

            # ************** plot error curve
            if epoch % self.print_res == 0:
                print("epoch {}: \t training accuracy: {:.5f}, (min: {:.5f})".format(
                    epoch, self.cost_train[epoch], np.min(self.cost_train[:epoch + 1])))
                self.errorcurves(epoch)
                self.plotprototype()
            # if 'fname' in kwargs.keys():
            #     self.save_results(kwargs['fname'])

    def errorcurves(self, epoch):
        nepochs = list(range(epoch+1))
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(nepochs, self.cost_train[nepochs], label='train set')
        ax.set_title('cost function')
        ax.set_xlabel('epoch')
        ax.set_ylabel('cost value')
        # plt.legend()
        plt.show()

    def plotprototype(self, isRGB = False):

        c = np.random.randint(self.nprotos, size=1)[0]

        ncols = 5
        nrows = self.xprotos[c].shape[1] // ncols
        fig, ax = plt.subplots(nrows, ncols, figsize=(15, 10))
        k = 0
        for i in range(nrows):
            for j in range(ncols):
                if isRGB:
                    m = np.min(self.xprotos[c, :, k])
                    M = np.max(self.xprotos[c, :, k])
                    pr = (self.xprotos[c, :, k] - m) / (M-m)
                    if nrows==1:
                        ax[j].imshow(
                            Vec2FIG(pr, isRGB=True),
                            vmin=self.xprotos[c, :, k].min(),
                            vmax=self.xprotos[c, :, k].max()
                        )
                    else:
                        ax[i, j].imshow(
                            Vec2FIG(pr, isRGB=True),
                            vmin=self.xprotos[c, :, k].min(),
                            vmax=self.xprotos[c, :, k].max()
                        )
                else:
                    if nrows==1:
                        ax[j].imshow(
                            Vec2FIG(self.xprotos[c, :, k], isRGB=False),
                            cmap=plt.get_cmap('gray'),
                            vmin=self.xprotos[c, :, k].min(),
                            vmax=self.xprotos[c, :, k].max()
                        )
                    else:
                        ax[i, j].imshow(
                            Vec2FIG(self.xprotos[c, :, k], isRGB=False),
                            cmap=plt.get_cmap('gray'),
                            vmin=self.xprotos[c, :, k].min(),
                            vmax=self.xprotos[c, :, k].max()
                        )
                k += 1
            #     if k==self.nprotos :
            #         break
            # if k==self.nprotos:
            #     break
        # plt.suptitle("class: {}".format(self.yprotos[c]), fontsize=30)
        plt.show()






def plotprototype(xprotos, dirname, idx):
    ncols = 5
    nrows = xprotos.shape[1] // ncols
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 10))
    d = int(np.sqrt(xprotos.shape[0]))
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            if nrows==1:
                ax[j].imshow(xprotos[:, k].reshape((d, d), order='C'),
                                cmap=plt.get_cmap('gray'),
                                vmin=xprotos[:, k].min(),
                                vmax=xprotos[:, k].max())
            else:
                ax[i, j].imshow(xprotos[:, k].reshape((d, d), order='C'),
                                cmap=plt.get_cmap('gray'),
                                vmin=xprotos[:, k].min(),
                                vmax=xprotos[:, k].max())
            k += 1

    path = os.path.join('figures', dirname)
    try:
        os.makedirs(path, exist_ok = True)
    except OSError as error:
        print("Directory can not be created" )
    filename = os.path.join(path, str(idx) + ".png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')

def load_results(fname):
    with np.load(fname + '.npz') as file:
        xprotos = file['xprotos']
    for i in range(xprotos.shape[0]):
        plotprototype(xprotos[i], fname, i)

if __name__ == '__main__':
    load_results('eth80')