import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
from numpyy.utils_model import *
from sklearn.metrics import confusion_matrix, accuracy_score# f1_score
import torch

def orthoganization_of_data(data_3d):
    for i, data in enumerate(data_3d):
        data_3d[i] = LA.orth(data)
    return data_3d


class Model():
    def __init__(self, xprotos, yprotos, metric_type, **kwargs):        
            
        if 'sigma' in kwargs.keys():
            self.sigma = kwargs['sigma']
        else:
            self.sigma = 1
        if 'actfun' in kwargs.keys():
            self.act_fun = kwargs['actfun']
        else:
            self.act_fun = 'sigmoid'
        
        self.metric_type = metric_type
        self.dim_of_data = xprotos.shape[-2]
        self.dim_of_subspace = xprotos.shape[-1]
        self.xprotos = xprotos
        self.yprotos = yprotos
        self.lamda = np.ones((1, self.dim_of_subspace)) / self.dim_of_subspace
        

        
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
        
        print('Qplus', results['Q'][iplus][:2])
        print('Qminus', results['Q'][iminus][:2])
        
        return plus, minus

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
        print(self.der_act_fun(cost), 2*dminus / ((dplus + dminus) ** 2))
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
        if self.metric_type == 'pseudo-chordal':
            return - canonicalcorrelation.T
        else:
            return np.arccos(canonicalcorrelation).T ** 2

    def fit(self, xdata, ydata, ** kwargs):
        """
        Perform one epoch of training using the provided training data.
        """        
        
        if xdata.shape==2:
            xdata.unsqueeze(0)
            
        Eucl_grad_plus_list =[]
        Eucl_grad_minus_list =[]
        Eucl_grad_lambda_list = []
        dE_dist_plus_list = []
        dE_dist_minus_list=[]
        proto_rot_plus_list = []
        proto_rot_minus_list = []
        Qminus_list = []
        Qplus_list = []
        Qwplus_list = []
        Qwminus_list = []
        for data, label in zip(xdata, ydata):
            plus, minus = self.findWinner(data, label)
            print('distances', plus['distance'], minus['distance'])
            cost = self.loss(plus['distance'], minus['distance'])

            # rotation of the coordinate system
            X_rot_plus = data @ plus['Q']
            X_rot_minus = data @ minus['Q']
            proto_rot_plus = self.xprotos[plus['index']] @ plus['Qw']
            proto_rot_minus = self.xprotos[minus['index']] @ minus['Qw']
            print('rotated')
            print(X_rot_plus[:2])
            print(X_rot_minus[:2])
            print(self.lamda)
            # Compute gradients
            dE_dist_plus = self.dE_distance_plus(cost, plus['distance'], minus['distance'])
            dE_dist_minus = self.dE_distance_minus(cost, plus['distance'], minus['distance'])

            print("derivative distances")
            print(dE_dist_plus, dE_dist_minus)
            Eucl_grad_plus = self.Euclidean_gradient(dE_dist_plus, X_rot_plus, plus['canonicalcorr'])
            Eucl_grad_minus = self.Euclidean_gradient(dE_dist_minus, X_rot_minus, minus['canonicalcorr'])

            print("iplus", plus['index'], "iminus", minus['index'])

            Eucl_grad_lambda = (            
                dE_dist_plus * self.der_distance_relevance(plus['canonicalcorr']) +
                dE_dist_minus * self.der_distance_relevance(minus['canonicalcorr'])            
            )
            
            Eucl_grad_plus_list.append(Eucl_grad_plus)
            Eucl_grad_minus_list.append(Eucl_grad_minus)
            Eucl_grad_lambda_list.append(Eucl_grad_lambda)
            dE_dist_plus_list.append(dE_dist_plus)
            dE_dist_minus_list.append(dE_dist_minus)
            proto_rot_plus_list.append(proto_rot_plus)
            proto_rot_minus_list.append(proto_rot_minus)
            Qplus_list.append(plus['Q'])
            Qminus_list.append(minus['Q'])
            Qwplus_list.append(plus['Qw'])
            Qwminus_list.append(minus['Qw'])
        
        Eucl_grad_plus_torch = torch.stack([torch.from_numpy(i) for i in Eucl_grad_plus_list], axis=0)
        Eucl_grad_minus_torch = torch.stack([torch.from_numpy(i) for i in Eucl_grad_minus_list], axis=0)
        Eucl_grad_lambda_torch = torch.stack([torch.from_numpy(i) for i in Eucl_grad_lambda_list], axis=0)
#         print(dE_dist_plus_list)
        dE_dist_plus_torch = torch.tensor(dE_dist_plus_list)
        dE_dist_minus_torch = torch.tensor(dE_dist_minus_list)
        proto_rot_plus_torch = torch.stack([torch.from_numpy(i) for i in proto_rot_plus_list], axis=0)
        proto_rot_minus_torch = torch.stack([torch.from_numpy(i) for i in proto_rot_minus_list], axis=0)
        Qplus_torch = torch.stack([torch.from_numpy(i) for i in Qplus_list], axis=0)
        Qminus_torch = torch.stack([torch.from_numpy(i) for i in Qminus_list], axis=0)
        Qwplus_torch = torch.stack([torch.from_numpy(i) for i in Qwplus_list], axis=0)
        Qwminus_torch = torch.stack([torch.from_numpy(i) for i in Qwminus_list], axis=0)
        
        return Eucl_grad_plus_torch, Eucl_grad_minus_torch, Eucl_grad_lambda_torch, dE_dist_plus_torch, dE_dist_minus_torch, proto_rot_plus_torch, proto_rot_minus_torch, Qplus_torch, Qminus_torch, Qwplus_torch, Qwminus_torch


def load_results(fname):
    with np.load(fname + '.npz') as file:
        xprotos = file['xprotos']
    # for i in range(xprotos.shape[0]):
    #     plotprototype(xprotos[i], fname, i)

# if __name__ == '__main__':
#     load_results('eth80')