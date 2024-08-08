"""
Modified Script - Original Source: https://github.com/M-Nauta/ProtoTree/tree/main

Description:
This script is a modified version of the original script by M. Nauta.
Modifications have been made to suit specific requirements or preferences.
"""

from tqdm import tqdm
import torch.utils.data
from torch.utils.data import DataLoader

from lvqmodel.model import Model
from util.glvq import metrics
from util.log import Log

import numpy as np


@torch.no_grad()
def eval(
        model: Model,
        test_loader: DataLoader,
        epoch: int,
        loss,
        device,
        log: Log = None,
        log_prefix: str = 'log_eval_epochs',
        progress_prefix: str = 'Eval Epoch'
) -> dict :

    model = model.to(device)

    # to store information about the procedure
    test_info = dict()

    model.eval()

    # to show the progress-bar
    train_iter = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc=progress_prefix + ' %s' % epoch,
        ncols=0
    )

    conf_mat = np.zeros((model._num_classes, model._num_classes), dtype=int)

    # training process (one epoch)
    for i, (xs, ys) in train_iter:

        xs, ys = xs.to(device), ys.to(device)

        # forward pass
        distances, _ = model.prototype_layer(xs)

        # predict labels
        yspred = model.prototype_layer.yprotos[distances.argmin(axis=1)]
        cost, iplus, iminus = loss(
            ys,
            model.prototype_layer.yprotos_mat,
            model.prototype_layer.yprotos_comp_mat,
            distances)

        # compute the confusion matrix
        acc, cmat = metrics(ys, yspred, nclasses=model._num_classes)
        conf_mat += cmat

        train_iter.set_postfix_str(
            f"Batch [{i + 1}/{len(test_loader)}, Loss: {cost.item(): .3f}, Acc: {acc: .3f}"
        )

    test_info['confusion_matrix'] = conf_mat
    test_info['test_accuracy'] = np.diag(conf_mat).sum() / conf_mat.sum()

    log.log_message("\nEpoch %i - Test accuracy: %s" %(epoch, test_info['test_accuracy']))

    return test_info
