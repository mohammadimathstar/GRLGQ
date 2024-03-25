from preprocessing_img import *
import numpy as np

from model import *
from joblib import Parallel, delayed


def executions():
    distance_type = 'pseudo-chordal'  # chordal or geodesic  pseudo-chordal
    dataname = 'ETH-80_highres'  # 'ETH-80_highres', 'YaleB_highres'
    num_of_epochs = 500
    subspace_dim = 5
    localized = False
    balanced = False
    nprotos = 1
    sigma = 100
    act_fun = 'identity'  # 'sigmoid', 'identity'

    which_fold = 0 # 0 to 9

    lr_w = 0.1  #
    lr_r = 0.0001  #

    data, labels, train_test_ids = read_matfile(dataname)

    Xtrain = data[train_test_ids[which_fold, 0]][0]
    Ytrain = labels[train_test_ids[which_fold, 0]][0]
    Xval = data[train_test_ids[which_fold, 1]][0]
    Yval = labels[train_test_ids[which_fold, 1]][0]

    Ytrain, Yval = np.squeeze(Ytrain), np.squeeze(Yval)

    if Xtrain.shape[-1] != subspace_dim:
        Xtrain = Xtrain[:, :, :subspace_dim]
        Xval = Xval[:, :, :subspace_dim]

    print(
        f"\nThere are '{Xtrain.shape[0]}' training and '{Xval.shape[0]}' testing examples on the manifold G({Xtrain.shape[-2]}, {Xtrain.shape[-1]}).")

    # ************** build the model **************
    # Note: for chordal distance, the learning rate (for w and r) should be
    # bigger than the case in geodesic distance
    print('\nConstructing the model ...')
    model = Model(
        dim_of_data=Xtrain.shape[-2],  # the dimensionality of data
        dim_of_subspace=Xtrain.shape[-1],  # number of data in a set
        num_of_classes=len(np.unique(Ytrain)),  # number of classes
        distance=distance_type,  # (pseudo) chordal or geodesic
        balanced=balanced,
        localized=localized,
        nprotos=nprotos,  # for now it is only 1: check if you need to modify it for more
        actfun=act_fun,  # the function inside cost function: identity or sigmoid
        sigma=sigma,  # parameter for sigmoid function
    )

    # ******* initialize prototypes ***********
    print('Initializing prototypes ...')
    # via samples
    # model.initialize_parameters(xtrain=Xtrain, ytrain=Ytrain)
    # via normal distributions
    model.initialize_parameters(classes=Ytrain)

    acc_train = np.zeros(num_of_epochs + 1)
    acc_val = np.zeros(num_of_epochs + 1)
    cost_train = np.zeros(num_of_epochs + 1)
    cost_val = np.zeros(num_of_epochs + 1)

    pred = model.prediction(Xtrain)
    acc_train[0], conf_mat_tr = model.metrics(Ytrain, pred)
    cost_train[0] = model.get_loss(Xtrain, Ytrain)

    if Xval.size != 0:
        pred = model.prediction(Xval)
        acc_val[0], conf_mat_val = model.metrics(Yval, pred)
        cost_val[0] = model.get_loss(Xval, Yval)
        print("epoch {}: \t training accuracy: {:.2f} - cost: {:.3f}, \t testing accuracy: {:.2f} - cost: {:.3f} (max: {:.5f})".format(
            0, acc_train[0], cost_train[0], acc_val[0], cost_val[0], acc_val[0]))
    else:
        print("epoch {}: \t accuracy: {:.2f}".format(
            0, acc_train[0]))
    # np.set_printoptions(precision=1)
    # print(conf_mat_tr)

    fname = "../model/%s/%s_model_d%i_%s" % (
        dataname.split("_")[0],
        dataname,
        Xtrain.shape[-1],
        distance_type[:2]
    )
    print('Fitting the model ...')
    for epoch in range(1, num_of_epochs + 1):
        model.fit(
            Xtrain, Ytrain,
            lr_w=lr_w, lr_r=lr_r,
        )
        if epoch % 10 == 0:
            lr_w *= 0.5

        pred_tr = model.prediction(Xtrain)
        acc_train[epoch], conf_mat_tr = model.metrics(Ytrain, pred_tr)

        pred_val = model.prediction(Xval)
        acc_val[epoch], conf_mat_val = model.metrics(Yval, pred_val)

        cost_train[epoch] = model.get_loss(Xtrain, Ytrain)
        cost_val[epoch] = model.get_loss(Xval, Yval)

        if epoch % 50 == 0:
            print("relevances: ", model.lamda)
            # print("epoch {}: \t training accuracy: {:.2f}, \t testing accuracy: {:.2f} (max: {:.5f})".format(
            #     epoch, acc_train[epoch], acc_val[epoch], np.max(acc_val[:epoch + 1])))
            print(
                "epoch {}: \t training accuracy: {:.2f} - cost: {:.3f}, \t testing accuracy: {:.2f} - cost: {:.3f} (max: {:.5f})".format(
                    epoch, acc_train[epoch], cost_train[epoch], acc_val[epoch], cost_val[epoch], acc_val[epoch]))
            np.set_printoptions(precision=1)
            print(conf_mat_val)

            errorcurves(acc_train=acc_train[:epoch + 1], acc_val=acc_val[:epoch + 1], lamda=model.lamda)
            print(f"save model in: %s" % fname)
            model.save_results(
                fname, acc_train[:epoch + 1], acc_val[:epoch + 1], conf_mat_val,
                cost_train[:epoch + 1], cost_val[:epoch + 1],
            )


if __name__ == '__main__':
    executions()