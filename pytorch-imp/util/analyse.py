import torch
import argparse
from torch.utils.data import DataLoader
from grlgq.model import Model
from util.log import Log


def analyse_output_shape(model: Model, trainloader: DataLoader, log: Log, device):
    with torch.no_grad():
        # Get a batch of training data
        xs, ys = next(iter(trainloader))
        xs, ys = xs.to(device), ys.to(device)
        log.log_message("Image input shape: "+str(xs[0,:,:,:].shape))
        log.log_message("Features output shape (without 1x1 conv layer): "+str(model._net(xs).shape))
        # log.log_message("Convolutional output shape (with 1x1 conv layer): "+str(model._add_on(model._net(xs)).shape))
        log.log_message("Prototypes shape: "+str(model.prototype_layer.xprotos.shape))
