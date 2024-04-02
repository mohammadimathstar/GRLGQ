import os
import argparse
import pickle
import numpy as np
import random
import torch
import torch.optim

"""
    Utility functions for handling parsed arguments

"""
def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser('Train a ProtoTree')
    parser.add_argument('--dataset',
                        type=str,
                        default='CUB-200-2011',
                        help='The name of dataset for training.')
    parser.add_argument('--net',
                        type=str,
                        default='resnet50_inat',
                        help='Base network used in the tree. Pretrained network on iNaturalist is only available for '
                             'resnet50_inat (default). Others are pretrained on ImageNet. Options are: resnet18, '
                             'resnet34, resnet50, resnet50_inat, resnet101, resnet152, densenet121, densenet169, '
                             'densenet201, densenet161, vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn or '
                             'vgg19_bn')
    parser.add_argument('--batch_size_train',
                        type=int,
                        default=8,
                        help='Batch size of training data.')
    parser.add_argument('--batch_size_test',
                        type=int,
                        default=64,
                        help='Batch size for computing test error.')
    parser.add_argument('--nepochs',
                        type=int,
                        default=100,
                        help='The number of epochs to train the prototypes.')
    parser.add_argument('--subspace_dim',
                        type=int,
                        default=10,
                        help="The dimensionality 'd' of subspaces on the Grassmann manifold G(D,d)."
                        )
    parser.add_argument('--cost_fun',
                        type=str,
                        default='identity',
                        help="The function used inside the error function: a) sigmoid, b) identity."
                        )
    parser.add_argument('--sigma',
                        type=int,
                        default=100,
                        help="The hyperparameter for sigmoid function (controling the slope)."
                        )
    parser.add_argument('--metric_type',
                        type=str,
                        default='pseudo-chordal',
                        help="The type of distance to use: a) geodesic, b) pseudo-chordal."
                        )
    parser.add_argument('--num_of_protos',
                        type=int,
                        default=1,
                        help="The number of prototypes per class."
                        )
    parser.add_argument('--dim_of_data',
                        type=int,
                        default= 7*7,
                        help="Number of pixels in the last layer of the feature net."
                        )
    parser.add_argument('--dim_of_subspace',
                        type=int,
                        default=10,
                        help="The dimensionality of subspaces 'd'."
                        )

    parser.add_argument('--lr',
                        type=float,
                        default=0.1,
                        help='The learning rate for the training of the prototypes')
    parser.add_argument('--lr_rel',
                        type=float,
                        default=0.0001,
                        help='The learning rate for the training of the relevances.')
    
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    parser.add_argument('--log_dir',
                        type=str,
                        default='./run_prototypes',
                        help='The directory in which train progress should be logged')
    
    parser.add_argument('--milestones',
                        type=str,
                        default='',
                        help='The milestones for the MultiStepLR learning rate scheduler')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.5,
                        help='The gamma for the MultiStepLR learning rate scheduler. Needs to be 0<=gamma<=1')
    parser.add_argument('--state_dict_dir_net',
                        type=str,
                        default='',
                        help='The directory containing a state dict with a pretrained backbone network')
    parser.add_argument('--state_dict_dir_tree',
                        type=str,
                        default='',
                        help='The directory containing a state dict (checkpoint) with a pretrained prototype. Note '
                             'that training further from a checkpoint does not seem to work correctly. Evaluating a '
                             'trained prototype does work.')
    parser.add_argument('--freeze_epochs',
                        type=int,
                        default=-1,
                        help='Number of epochs where pretrained features_net will be frozen.'
                        )
    parser.add_argument('--dir_for_saving_images',
                        type=str,
                        default='upsampling_results',
                        help='Directoy for saving the prototypes, patches and heatmaps.')
    parser.add_argument('--upsample_threshold',
                        type=float,
                        default=0.98,
                        help='Threshold (between 0 and 1) for visualizing the nearest patch of an image after '
                             'upsampling. The higher this threshold, the larger the patches.')
    parser.add_argument('--disable_pretrained',
                        action='store_true',
                        help='When set, the backbone network is initialized with random weights instead of being '
                             'pretrained on another dataset). When not set, resnet50_inat is initalized with weights '
                             'from iNaturalist2017. Other networks are initialized with weights from ImageNet'
                        )
    parser.add_argument('--kontschieder_train',
                        action='store_true',
                        help='Flag that first trains the leaves for one epoch, and then trains the rest of ProtoTree '
                             '(instead of interleaving leaf and other updates). Computationally more expensive.'
                        )
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



if __name__ == '__main__':
    ar = get_args()
    save_args(ar, './')


