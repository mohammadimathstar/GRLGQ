"""
Modified Script - Original Source: https://github.com/M-Nauta/ProtoTree/tree/main

Description:
This script is a modified version of the original script by M. Nauta.
Modifications have been made to suit specific requirements or preferences.

"""

import os
import argparse
import pickle


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser('Train a PrototypeLayer.')
    parser.add_argument('--dataset',
                        type=str,
                        default='ETH-80',
                        help='The name of dataset for training.')
    parser.add_argument('--batch_size_train',
                        type=int,
                        default=1,
                        help='Batch size of training data.')
    parser.add_argument('--batch_size_test',
                        type=int,
                        default=32,
                        help='Batch size for computing test error.')
    parser.add_argument('--nepochs',
                        type=int,
                        default=100,
                        help='The number of epochs to train the prototypes.')
    parser.add_argument('--cost_fun',
                        type=str,
                        default='identity',
                        help="The function (mu) used inside the cost function: "
                             "a) sigmoid, b) identity, c) relu, d) leaky_relu, e) elu, f) rrelu."
                        )
    parser.add_argument('--sigma',
                        type=int,
                        default=None,
                        help="The hyperparameter for the cost function."
                        )
    parser.add_argument('--metric_type',
                        type=str,
                        default='geodesic',
                        help="The type of distance to use: a) geodesic, b) chordal."
                        )
    parser.add_argument('--num_of_protos',
                        type=int,
                        default=1,
                        help="The number of prototypes per class."
                        )
    parser.add_argument('--dim_of_subspace',
                        type=int,
                        default=5,
                        help="The dimensionality of subspaces 'd'."
                        )
    parser.add_argument('--lr_protos',
                        type=float,
                        default=0.01,
                        help='The learning rate for the training of the prototypes')
    parser.add_argument('--lr_rel',
                        type=float,
                        default=0.0001,
                        help='The learning rate for the training of the relevances.')
    parser.add_argument('--milestones',
                        type=str,
                        default='',
                        help='The milestones for the MultiStepLR learning rate scheduler')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.5,
                        help='The gamma for the MultiStepLR learning rate scheduler. Needs to be 0<=gamma<=1')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    parser.add_argument('--log_dir',
                        type=str,
                        default='./run_prototypes',
                        help='The directory in which train progress should be logged')
    parser.add_argument('--state_dict_dir_model',
                        type=str,
                        default='',
                        help='The directory containing a state dict (checkpoint) with a pretrained prototype. Note '
                             'that training further from a checkpoint does not seem to work correctly. Evaluating a '
                             'trained prototype does work.')

    args = parser.parse_args()
    args.milestones = get_milestones(args)
    return args


def get_milestones(args: argparse.Namespace):
    """
    Parse the milestones argument to get a list
    :param args: The arguments given
    """
    if args.milestones != '':
        milestones_list = args.milestones.split(',')
        for m in range(len(milestones_list)):
            milestones_list[m]=int(milestones_list[m])
    else:
        milestones_list = []
    return milestones_list


def save_args(args: argparse.Namespace, directory_path: str) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exists, create it
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + '/args.txt', 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write('{}: {}\n'.format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + '/args.pickle', 'wb') as f:
        pickle.dump(args, f)                                                                               
    

def load_args(directory_path: str) -> argparse.Namespace:
    """
    Load the pickled arguments from the specified directory
    :param directory_path: The path to the directory from which the arguments should be loaded
    :return: the unpickled arguments
    """
    with open(directory_path + '/args.pickle', 'rb') as f:
        args = pickle.load(f)
    return args

