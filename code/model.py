import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import linalg as LA
import os
from preprocessing import *

# TODO: adding more advance optimizer (e.g. adaptive lr)


def orthoganization_of_data(data_3d):
    for i, data in enumerate(data_3d):
        data_3d[i] = LA.orth(data)
    return data_3d

class Model():
    def __init__(self, **kwargs):
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

        self.acc_train = np.zeros(self.nepochs+1)
        self.cost_train = np.zeros(self.nepochs+1)
        self.acc_val = np.zeros(self.nepochs+1)
        self.cost_val = np.zeros(self.nepochs+1)

    def init_protos_random(self, labels):
        """
        use gaussian distribution to initialize prototypes
        :return: initial value for prototypes
        """
        self.xprotos = np.random.normal(
            0, 1,
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
        from scipy import linalg as LA

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
            return (d_plus - d_minus) / (d_plus + d_minus)
        else:
            return self.sigmoid((d_plus - d_minus) / (d_plus + d_minus))


    def get_distances_to_prototypes(self, xdata):
        """
        it computes the chordal distance between two subspaces (using canonical correlation)
        :param xdata: a subspace representing by an orthonormal matrix of size (D * d)
        :param type: chordal or pseudo chordal
        :return: canonical correlation, chordal distance and principal directions
        """

        U, S, Vh = np.linalg.svd(
            xdata.T @ self.xprotos,
            full_matrices=False,
            compute_uv=True,
            hermitian=False # If True, a is assumed to be Hermitian (symmetric if real-valued)
        )

        output = dict()
        output['principal_1'] = U
        output['principal_2'] = np.transpose(Vh, (0, 2, 1))
        output['canonicalcorrelation'] = S

        if self.distance =='chordal':
            output['distance'] = np.sum(self.lamda) - (self.lamda @ (S.T**2)).T
        else:
            output['distance'] = np.sum(self.lamda) - (self.lamda @ (np.arccos(S).T ** 2)).T

        assert np.sum(output['distance'] < 0) < 1, "distance is negative"
        return output

    def findWinner(self, data, label):
        """
        It find the closest prototypes to datapoint with the same/different labels
        :param data: a subspace represented by an orthonormal matrix (D * d)
        :param label: the label of the data point, an integer number
        :return: the closest protos with the same/different labels (plus, minus)
        """
        results = self.get_distances_to_prototypes(data)

        sameclass = np.argwhere(self.yprotos == label).T[0]
        diffclass = np.argwhere(self.yprotos != label).T[0]

        iplus = sameclass[np.argmin(results['distance'][sameclass])]
        iminus = diffclass[np.argmin(results['distance'][diffclass])]
        dplus = results['distance'][iplus]
        dminus = results['distance'][iminus]

        plus = dict()
        plus['index'] = iplus
        plus['distance'] = dplus
        plus['Q'] = results['principal_1'][iplus]
        plus['Qw'] = results['principal_2'][iplus]
        plus['canonicalcorr'] = results['canonicalcorrelation'][iplus]

        minus = dict()
        minus['index'] = iminus
        minus['distance'] = dminus
        minus['Q'] = results['principal_1'][iminus]
        minus['Qw'] = results['principal_2'][iminus]
        minus['canonicalcorr'] = results['canonicalcorrelation'][iminus]
        return plus, minus

    def evaluate(self, data, labels):
        acc = 0; cost = 0
        for xdata, ydata in zip(data, labels):
            plus, minus = self.findWinner(xdata, ydata)
            cost += self.loss(plus['distance'], minus['distance'])[0]
            if plus['distance'] < minus['distance']:
                acc += 1
        acc /= (len(labels) / 100)
        cost /= len(labels)

        return acc, cost

    # ***********************************
    # ***** computing derivatives *******
    # ***********************************

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

    def dE_distance_plus(self, cost, dplus, dminus):
        """
        it computes the derivative of error function w.r.t. distance to W^+ (winner prototype with the same label)
        :param cost: the amount of cost caused by the sample
        :param dplus: distance between the sample and W^+
        :param dminus: distance between the sample and W^-
        :return: dE / dD^+
        """
        return 2 * self.der_act_fun(cost) * dminus / ((dplus + dminus) ** 2)

    def dE_distance_minus(self, cost, dplus, dminus):
        """
        it computes the derivative of error function w.r.t. distance to W^- (winner prototype with a different label)
        :param cost: the amount of cost caused by the sample
        :param dplus: distance between the sample and W^+
        :param dminus: distance between the sample and W^-
        :return: dE / dD^-
        """
        return - 2 * self.der_act_fun(cost) * dplus / ((dplus + dminus) ** 2)

    def der_W_chordal(self, X_rotated, canonicalcorrelation):
        """
        it computes the derivative of distance (sum_i r_i * sin^2(theta_i)) w.r.t W (the winner prototype)
        :param X_rotated:  principal direction
        :param canonicalcorrelation: canonical correlation (i.e. cos(t))
        :return: d D / dW
        """
        Lam = np.tile(
            self.lamda[0] * canonicalcorrelation,
            (self.dim_of_data, 1)
        )
        return - 2 * Lam * X_rotated


    def der_W_geodesic(self, X_rotated, canonicalcorrelation):
        """
        it computes the derivative of distance (sum_i r_i * theta_i^2) w.r.t W (the winner prototype)
        :param X_rotated: principal direction
        :param canonicalcorrelation: canonical correlation (i.e. cos(t))
        :return: d D / dW
        """
        G = 2 * np.diag(
            self.lamda[0] * np.arccos(canonicalcorrelation) / np.sqrt(1 - canonicalcorrelation ** 2)
        )
        return - X_rotated @ G

    def Euclidean_gradient(self, dE_dist, X_rot, CC):
        """
        it computes the (euclidean) derivative of the error function w.r.t. the winner prototype
        :param dE_dist: derivative of the error function w.r.t. distance to the winner prototype
        :param X_rot: principal direction of the sample
        :param CC: canonical correlation i.e. cos(theta)
        :return:
        """
        if self.distance == 'chordal':
            return dE_dist * self.der_W_chordal(X_rot, CC)
        else:
            return dE_dist * self.der_W_geodesic(X_rot, CC)


    def der_distance_relevance(self, canonicalcorrelation):
        """
        it computes the derivative of distance w.r.t relevance factors i.e. lambda
        :param canonicalcorrelation: canonical correlation i.e. cos(theta)
        :return:
        """
        if self.distance == 'chordal':
            return - (canonicalcorrelation.T ** 2)
        else:
            return np.arccos(canonicalcorrelation).T ** 2


    def one_epoch(self, xtrain, ytrain, epoch, low_bound_lambda=0.001):
        from scipy import linalg as LA

        for i in np.random.permutation(xtrain.shape[0]):
            plus, minus = self.findWinner(xtrain[i], ytrain[i])
            cost = self.loss(plus['distance'], minus['distance'])[0]

            # rotation of the coordinate system
            X_rot_plus = xtrain[i] @ plus['Q']; X_rot_minus = xtrain[i] @ minus['Q']
            proto_rot_plus = self.xprotos[plus['index']] @ plus['Qw']
            proto_rot_minus = self.xprotos[minus['index']] @ minus['Qw']

            # ************************************
            # ******** compute gradients *********
            # ************************************
            dE_dist_plus = self.dE_distance_plus(cost, plus['distance'][0], minus['distance'][0])
            dE_dist_minus = self.dE_distance_minus(cost, plus['distance'][0], minus['distance'][0])

            Eucl_grad_plus = self.Euclidean_gradient(dE_dist_plus, X_rot_plus, plus['canonicalcorr'])
            Eucl_grad_minus = self.Euclidean_gradient(dE_dist_minus, X_rot_minus, minus['canonicalcorr'])

            # ************************************
            # ******** update prototypes *********
            # ************************************

            self.xprotos[plus['index']] = proto_rot_plus - self.lr_w * Eucl_grad_plus
            self.xprotos[minus['index']] = proto_rot_minus - self.lr_w * Eucl_grad_minus

            # orthonormalization prototypes
            self.xprotos[plus['index']] = LA.orth(self.xprotos[plus['index']])
            self.xprotos[minus['index']] = LA.orth(self.xprotos[minus['index']])

            # *******************************************
            # ******** update relevance factors *********
            # *******************************************

            self.lamda[0] -= self.lr_r * (
                dE_dist_plus * self.der_distance_relevance(plus['canonicalcorr']) + \
                dE_dist_minus * self.der_distance_relevance(minus['canonicalcorr'])
            )

            # normalization of relevance factors
            self.lamda[0, np.argwhere(self.lamda < low_bound_lambda)[:,1]] = low_bound_lambda
            self.lamda[0] = self.lamda[0] / np.sum(self.lamda)

    def initialize_prototypes(self, xtrain, ytrain):
        self.lamda = np.ones((1, self.dim_of_subspace)) / self.dim_of_subspace

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


    def fit(self, xtrain, ytrain, **kwargs):
        if 'plot_res' in kwargs.keys():
            plot_res = kwargs['plot_res']
        else:
            plot_res = None

        if 'initmethod' not in kwargs.keys():
            self.initmethod = 'pca'
        else:
            self.initmethod = kwargs['initmethod']

        self.number_of_class = len(np.unique(ytrain))
        self.dim_of_data = xtrain.shape[1]
        self.dim_of_subspace = xtrain.shape[2]

        if 'xval' in kwargs.keys():
            xval = kwargs['xval']
            yval = kwargs['yval']
        else:
            xval = np.array([])
            yval = np.array([])
            self.acc_val = np.array([])
            self.cost_val = np.array([])

        # ******** Initializing prototypes *********
        self.initialize_prototypes(xtrain, ytrain)

        self.acc_train[0], self.cost_train[0] = self.evaluate(xtrain, ytrain)
        if xval.size != 0:
            self.acc_val[0], self.cost_val[0] = self.evaluate(xval, yval)

        # ************** training *****************
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
            if plot_res is not None and (epoch % plot_res == 0):
                self.errorcurves(epoch)
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