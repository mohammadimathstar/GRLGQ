# from keras.datasets import mnist, fashion_mnist
# from tensorflow.keras.datasets import cifar10, cifar100
from skimage import color
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy import linalg as LA
import os
import imageio
import glob
import itertools
from scipy.io import savemat, loadmat
import cv2


def RGB2GRAY(X):
    Y = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    for i in range(X.shape[0]):
        Y[i] = color.rgb2gray(X[i])
    return Y

def RGB2GRAY2Vec(X):
    D = X.shape[1] * X.shape[2]
    Y = np.zeros((X.shape[0], D))
    for i in range(X.shape[0]):
        Y[i] = color.rgb2gray(X[i]).reshape(D, order='C')
    return Y

def FIG2Vec(X, D):
    return X.reshape(D, order='C')

def FIGs2Vecs(X, isRGB=True):
    """
    :param X: a gray-scale or rgb image
    :return:
    """
    if len(X.shape)==4:
        D = X.shape[1] * X.shape[2] * 3
    else:
        D = X.shape[1] * X.shape[2]
    # if len(X.shape)==3:
    #     D = X.shape[1] * X.shape[2]
    # elif (~isRGB):
    #     D = X.shape[1] * X.shape[2]
    # elif len(X.shape)==4:
    #     D = X.shape[1] * X.shape[2] * 3
    # else:
    #     raise Exception('data is not image!')

    if (len(X.shape)==4) & (~isRGB):
        # print('h',(len(X.shape)==4) and (~isRGB), 'h')
        Y = RGB2GRAY2Vec(X)
    else:
        Y = np.zeros((X.shape[0], D))
        for i in range(X.shape[0]):
            Y[i] = FIG2Vec(X[i], D)
    return Y

def Vec2FIG(X, isRGB=True):
    D = X.shape[0]
    if isRGB:
        m = int(np.sqrt(D // 3))
        return X.reshape((m, m, 3), order='C')
    else:
        m = n = int(np.sqrt(D))
        # m = 38
        # n = 24
        return X.reshape((m, n), order='C')#C or F


def LocalPCA(X, n_comp=5):
    """
    compute PCA for a set of samples
    :param X: list of samples (n * D)
    :return: eigen values and eiven vectors
    """
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    return pca.components_

def convert_random(X, y, d):
    N, D = X.shape
    Xset = np.zeros((N // d, D, d))
    yset = np.zeros(N // d)
    k = 0
    for label in range(len(np.unique(y))):
        idx = np.argwhere(y == label).T[0]
        idx = idx[np.random.permutation(len(idx))]
        for i in range(len(idx) // d):
            for j in range(d):
                Xset[k, :, j] = X[idx[i * d + j]]
            yset[k] = label
            Xset[k] = LA.orth(Xset[k])
            k += 1

    return Xset, yset

def convert_knn(X, y, setsize):
    """
    transforms a set of vectors into a collection of set of vectors
    :param X: [x_1; x_2; ..., x_N] rows contain feature vectors
    :param y: [y_1, y_2, ..., y_N] contains labels of vectors
    :param setsize: number of vectors inside a set
    :return: a 3-D array containing collections of 2-D arrays
    """
    N, D = X.shape
    Xset = np.zeros((N, D, setsize))
    yset = np.zeros(N)
    k = 0
    for l in range(len(np.unique(y))):
        idx = np.argwhere(y == l).T[0]
        n = idx.shape[0]
        data = X[idx]
        nbrs = NearestNeighbors(n_neighbors=setsize, algorithm='ball_tree').fit(data)
        ids = nbrs.kneighbors(data, return_distance=False)
        for i in range(n):
            Xset[k] = data[ids[i,:],:].T
            Xset[k] = LA.orth(Xset[k])
            yset[k] = l
            k += 1
    perm = np.random.permutation(N)
    Xset = Xset[perm]
    yset = yset[perm]
    return Xset, yset

def convert_random_knn(X, y, setsize, num_of_set_per_class):
    """
    transforms a set of vectors into a collection of set of vectors
    :param X: [x_1; x_2; ..., x_N] rows contain feature vectors
    :param y: [y_1, y_2, ..., y_N] contains labels of vectors
    :param setsize: number of vectors inside a set
    :return: a 3-D array containing collections of 2-D arrays
    """
    N = num_of_set_per_class * len(np.unique(y))
    _, D = X.shape
    Xset = np.zeros((N, D, setsize))
    yset = np.zeros(N)
    k = 0
    for l in range(len(np.unique(y))):
        idx = np.argwhere(y == l).T[0]
        # n = idx.shape[0]
        data_subset = X[np.random.permutation(idx)[:num_of_set_per_class]]
        data = X[idx]
        nbrs = NearestNeighbors(n_neighbors=setsize, algorithm='ball_tree').fit(data)
        ids = nbrs.kneighbors(data_subset, return_distance=False)
        for i in range(num_of_set_per_class):
            Xset[k] = data[ids[i,:],:].T
            Xset[k] = LA.orth(Xset[k])
            yset[k] = l
            k += 1
    perm = np.random.permutation(N)
    Xset = Xset[perm]
    yset = yset[perm]
    return Xset, yset


def load_data(dataname, n_comp=5, test_rate=0.2, isRGB=True):
    if dataname=='mnist':
        (xtrain, ytrain), (xval, yval) = mnist.load_data()
    elif dataname=='fashionmnist':
        (xtrain, ytrain), (xval, yval) = fashion_mnist.load_data()
    elif dataname=='cifar10':
        (xtrain, ytrain), (xval, yval) = cifar10.load_data()
    elif dataname=='cifar100':
        (xtrain, ytrain), (xval, yval) = cifar100.load_data()
    elif dataname=='Ex-Yale-faces':
        (xtrain, ytrain), (xval, yval) = load_ExtendedYaleFaces(test_rate=test_rate, imsize=(20, 20))
    elif dataname=='ETH80':
        xtrain, xval, ytrain, yval = load_ETH80(n_comp=n_comp, test_rate=test_rate, isRGB=isRGB, imsize=20)
    elif dataname == 'YTC':
        dataset, labels = load_YTC(n_comp=n_comp, test_rate=test_rate)
        save_data_set(dataset, labels)
        xtrain, xval, ytrain, yval = split_train_test(dataset, labels, test_rate)
    else:
        raise Exception("Sorry, the data set is not available")

    # *************** Converting 2/3-D images into 1-D vectors
    if dataname not in ('ETH80', 'YTC'):
        xtrain = FIGs2Vecs(xtrain, isRGB=isRGB,)
        xval = FIGs2Vecs(xval, isRGB=isRGB,)
        xtrain, ytrain = convert_random(xtrain, ytrain, n_comp)
        # xtrain, ytrain = convert_random_knn(xtrain, ytrain, n_comp, num_of_set_per_class=500)
        # xtrain, ytrain = convert_knn(xtrain, ytrain, n_comp)
        # xval, yval = convert_random(xval, yval, n_comp)
    else:
        save_train_test(xtrain, ytrain, xval, yval)

    return xtrain, ytrain, xval, yval

def save_data_set(X, y):
    mdic= {"data": X, "label": y}
    savemat("data_set.mat", mdic)

def save_train_test(xtrain, ytrain, xtest, ytest):
    mdic= {"xtrain": xtrain, "ytrain": ytrain, "xtest": xtest, "ytest": ytest}
    savemat("traintest.mat", mdic)

def load_ETH80(n_comp=5, test_rate=0.2, isRGB=True, imsize=256):
    parent_dir = "../dataset/ETH-80/"
    if isRGB:
        dataset = np.zeros((80, n_comp, imsize * imsize * 3))
    else:
        dataset = np.zeros((80, n_comp, imsize * imsize))

    labels = np.zeros(80)
    subset = np.zeros((41, imsize, imsize, 3))

    t = 0
    for label, set_i in itertools.product(range(1,9), range(1,11)):
        files = glob.glob(parent_dir + str(label) + "/" + str(set_i) + "/*.png")
        for (i, File) in enumerate(files):
            tmp = imageio.imread(File)
            subset[i] = cv2.resize(tmp, (imsize, imsize), interpolation=cv2.INTER_CUBIC)
        dataset[t] = LocalPCA(FIGs2Vecs(subset, isRGB), n_comp)
        labels[t] = label
        t += 1
    dataset = np.moveaxis(dataset, 2, 1)
    xtrain, xtest, ytrain, ytest = split_train_test(dataset, labels, test_rate)
    return xtrain, xtest, ytrain, ytest

def load_ExtendedYaleFaces(test_rate=0.3, imsize=(20, 20)):
    parent_dir = "../dataset/ExtendedYaleB/CroppedYale/"

    dataset = []
    labels = []
    train_ids = []
    test_ids = []
    t = 0
    for (i, person) in enumerate(os.listdir(parent_dir)):
        file_dir = glob.glob(parent_dir + person + "/*.pgm")
        num_train = int(len(file_dir) * (1-test_rate))
        r = np.random.permutation(len(file_dir))
        for (j, File) in enumerate(file_dir):
            im = imageio.imread(File)
            resized = cv2.resize(im, imsize, interpolation=cv2.INTER_CUBIC)
            dataset.append(resized)
            labels.append(i)

            if j in r[:num_train]:
                train_ids.append(t)
            else:
                test_ids.append(t)
            t = t+1

    dataset = np.array(dataset)
    labels = np.array(labels)
    return (dataset[train_ids], labels[train_ids]), (dataset[test_ids], labels[test_ids])

def get_ExtendedYaleFaces(imsize=(20, 20)):
    parent_dir = "../dataset/ExtendedYaleB/CroppedYale/"

    dataset = []
    labels = []
    for (i, person) in enumerate(os.listdir(parent_dir)):
        file_dir = glob.glob(parent_dir + person + "/*.pgm")
        for (j, File) in enumerate(file_dir):
            im = imageio.imread(File)
            resized = cv2.resize(im, imsize, interpolation=cv2.INTER_CUBIC)
            dataset.append(resized)
            labels.append(i)

    dataset = np.array(dataset)
    labels = np.array(labels)
    return dataset, labels


def load_YTC(n_comp=5, test_rate=0.2, min_num_images=20):
    if min_num_images<n_comp:
        min_num_images = n_comp
    parent_dir = "./myfigures/"
    data_list = []
    labels = [] #np.zeros(80)
    for (i, celebrity_name) in enumerate(os.listdir(parent_dir)):
        gallary_set = os.listdir(parent_dir + celebrity_name)
        for gallary_idx in gallary_set:
            file_dir = glob.glob(parent_dir + celebrity_name + "/" + gallary_idx + "/*.png")
            tmp_list = []
            for File in file_dir:
                tmp_list.append(cv2.equalizeHist(imageio.imread(File)))
            tmp = np.array(tmp_list)
            if tmp.shape[0] >= min_num_images:
                data_list.append(LocalPCA(FIGs2Vecs(tmp, isRGB=False), n_comp))
                labels.append(i)
    dataset = np.array(data_list)
    labels = np.array(labels)
    dataset = np.moveaxis(dataset, 2, 1)
    return dataset, labels


def split_train_test(X, y, test_rate=0.2):
    train_list, ytrain_list = list(), list()
    test_list, ytrain_list = list(), list()
    for label in np.unique(y):
        idx = np.argwhere(y==label).T[0]
        train_size = int((1-test_rate) * len(idx))
        p = np.random.permutation(idx)
        # print(idx.shape, train_size, p.shape)
        train_list.extend(p[:train_size])
        test_list.extend(p[train_size:])

    xtrain = X[train_list]
    ytrain = y[train_list]
    xtest = X[test_list]
    ytest = y[test_list]
    return xtrain, xtest, ytrain, ytest

def fivefoldCV():#X, y):
    with np.load('face_dataset.npz', allow_pickle=True) as file:
        X = file['data']
        y = file['label']
    train_list = [[],[],[],[],[]]
    test_list = [[],[],[],[],[]]
    train = [[],[],[],[],[]]
    test = [[], [], [], [], []]
    for label in np.unique(y):
        idx = np.argwhere(y==label).T[0]
        nrep = 45 // len(idx)
        if nrep<= 1:
            p = np.random.permutation(idx)[:45]
        else:
            p = np.random.permutation(idx)
            for i in range(nrep):
                p = np.append(p, np.random.permutation(idx))
            p = p[:45]
        for i in range(5):
            start_idx = i*9
            end_idx = (i+1)*9
            train_list[i].extend(p[start_idx:end_idx-3])
            test_list[i].extend(p[end_idx-3:end_idx])

    for i in range(5):
        train[i].append(X[train_list[i]])
        train[i].append(y[train_list[i]])
        test[i].append(X[test_list[i]])
        test[i].append(y[test_list[i]])
    np.savez('face', train=train, test=test)
    # return train, test

def load_5fold(fname):
    with np.load(fname + '.npz', allow_pickle=True) as file:
        train = file['train']
        test = file['test']
    return train, test

def read_matfile(filename="ETH-80", **kwargs):
    if 'traintestsplitfile' in kwargs.keys():
        train_test_idx_file = kwargs['traintestsplitfile']
    else:
        train_test_idx_file = filename
    matfile_dir = "../matlab/"
    f = loadmat(matfile_dir + filename + ".mat")
    data = f['dataset']
    labels = f['labels']
    g = loadmat(matfile_dir + train_test_idx_file + "_traintest.mat")
    data_split = g['ids']

    return data, labels, data_split


if __name__ == '__main__':
    # load_ETH80()
    # load_YTC()
    # load_data('mnist')
    load_data("Ex-Yale-faces", n_comp=7, test_rate=0.3)
    # load_ExtendedYaleFaces()
    # dataset, data_split = read_matfile()
    # read_matfile()
    # print(data_split)
