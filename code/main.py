from preprocessing import *
from model import *


def executions():
    dim_of_subspace = 7        # the dimensionality of the subspace
    distance_type = 'geodesic'  # (pseudo) chordal or geodesic
    isRGB = False
    dataname = 'Ex-Yale-faces'

    acc = np.zeros(10)
    acc_end = np.zeros(10)
    for i in range(10):
        print('Constructing the training/testing sets ...')
        Xtrain, Ytrain, Xval, Yval = load_data(dataname, dim_of_subspace, test_rate=0.25, isRGB=isRGB)

        # ************** build the model **************
        # Note: for chordal distance, the learning rate (for w and r) should be
        # bigger than the case in geodesic distance
        print('Constructing the model ...')
        model = \
            Model(#Localized(
                dim=Xtrain.shape[1],        # the dimensionality of data
                dim_of_subspace=dim_of_subspace,    # number of data in a set
                numclasses=len(np.unique(Ytrain)),  # number of classes
                distance=distance_type,     # (pseudo) chordal or geodesic
                lr_w=0.01,                  # learning rate for prototypes
                lr_r=0.00001,                # learning rate for relevance vector (smaller than lr_w)
                decay_rate=0.8,            # decay rate for learning rates
                regularizer_coef=0.00,      # coefficient of the regularization term
                maxepochs=500,   #250 yale          # maximum number of epochs
                nprotos=1,          # for now it is only 1: check if you need to modify it for more
                actfun = 'identity',# the function inside cost function: identity or sigmoid
                sigma=1,            # parameter for sigmoid function
                print_res=20,
            )

        # ************** fit the model **************
        print('Fitting the model ...')
        model.fit(
            Xtrain, Ytrain,         # training set (a collection of sets)
            xval=Xval, yval=Yval,   # testing set (a collection of sets)
            initmethod='pca',  # 'random', or 'pca' or 'samples' (it has problem)
            dataname=dataname,
            isRGB=isRGB,            # is the images RGB?
            fname=dataname          # the output file (model details)
        )
        acc[i] = np.max(model.acc_val)
        acc_end[i] = model.acc_val[-1]

        print(acc)
        print(np.mean(acc), np.std(acc))
        print(acc_end)
        print(np.mean(acc_end), np.std(acc_end))



if __name__ == '__main__':
    executions()
