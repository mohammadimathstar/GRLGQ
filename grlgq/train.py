"""
Modified Script - Original Source: https://github.com/M-Nauta/ProtoTree/tree/main

Description:
This script is a modified version of the original script by M. Nauta.
Modifications have been made to suit specific requirements or preferences.

"""

from tqdm import tqdm
import argparse

import torch
import torch.utils.data
import torch.utils.data
from torch.utils.data import DataLoader

from grlgq.model import Model
from grlgq.prototypes import rotate_prototypes
from util.grassmann import orthogonalize_batch
from util.log import Log

def train_epoch(
        model: Model,
        train_loader: DataLoader,
        epoch: int,
        loss,
        args: argparse.Namespace,
        optimizer_protos: torch.optim.Optimizer,
        optimizer_rel: torch.optim.Optimizer,
        device,
        log: Log = None,
        log_prefix: str = 'log_train_epochs',
        progress_prefix: str = 'Train Epoch'
) -> dict:

    model = model.to(device)

    # to store information about the procedure
    train_info = dict()
    total_loss = 0
    total_acc = 0

    # create a log
    log_loss = f"{log_prefix}_losses"

    # to show the progress-bar
    train_iter = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=progress_prefix + ' %s' % epoch,
        ncols=0
    )

    # training process (one epoch)
    for i, (xtrain, ytrain) in enumerate(train_loader):
        # ****** for the first solution
        optimizer_protos.zero_grad()
        optimizer_rel.zero_grad()

        xtrain, ytrain = xtrain.to(device), ytrain.to(device)
        distances, Qw = model.prototype_layer(xtrain)
        cost, iplus, iminus = loss(
            ytrain,
            model.prototype_layer.yprotos_mat,
            model.prototype_layer.yprotos_comp_mat,
            distances)

        cost.backward()

        ##### First way: using optimizers ##############
        with torch.no_grad():
            winners_ids, _ = torch.stack([iplus, iminus], axis=1).sort(axis=1)
            rotated_proto1, rotated_proto2 = rotate_prototypes(model.prototype_layer.xprotos, Qw, winners_ids)
            model.prototype_layer.xprotos[winners_ids[torch.arange(args.batch_size_train), 0]] = rotated_proto1
            model.prototype_layer.xprotos[winners_ids[torch.arange(args.batch_size_train), 1]] = rotated_proto2

        optimizer_protos.step()
        optimizer_rel.step()

        with torch.no_grad():
            model.prototype_layer.xprotos[winners_ids[torch.arange(args.batch_size_train), 0]] = orthogonalize_batch(
                model.prototype_layer.xprotos[winners_ids[torch.arange(args.batch_size_train), 0]]
            )
            model.prototype_layer.xprotos[winners_ids[torch.arange(args.batch_size_train), 1]] = orthogonalize_batch(
                model.prototype_layer.xprotos[winners_ids[torch.arange(args.batch_size_train), 1]]
            )
            #CHECK
            LOW_BOUND_LAMBDA = 0.0001
            model.prototype_layer.relevances[0, torch.argwhere(model.prototype_layer.relevances < LOW_BOUND_LAMBDA)[:, 1]] = LOW_BOUND_LAMBDA
            model.prototype_layer.relevances[:] = model.prototype_layer.relevances[
                                                  :] / model.prototype_layer.relevances.sum()

        ### ************ second way: manually update
        # with torch.no_grad():
        #     winners_ids, _ = torch.stack([iplus, iminus], axis=1).sort(axis=1)
        #     rotated_proto1, rotated_proto2 = rotate_prototypes(model.prototype_layer.xprotos, Qw, winners_ids)
        #     update1 = rotated_proto1 - args.lr_protos * model.prototype_layer.xprotos.grad[winners_ids[torch.arange(args.batch_size_train), 0]]
        #     update2 = rotated_proto2 - args.lr_protos * model.prototype_layer.xprotos.grad[winners_ids[torch.arange(args.batch_size_train), 1]]
        #     model.prototype_layer.xprotos[winners_ids[torch.arange(args.batch_size_train), 0]] = orthogonalize_batch(update1)
        #     model.prototype_layer.xprotos[winners_ids[torch.arange(args.batch_size_train), 1]] = orthogonalize_batch(update2)
        #     model.prototype_layer.relevances -= args.lr_rel * model.prototype_layer.relevances.grad / args.batch_size_train
            # CHECK
            # LOW_BOUND_LAMBDA = 0.0001
            # model.prototype_layer.relevances[
            #     0, torch.argwhere(model.prototype_layer.relevances < LOW_BOUND_LAMBDA)[:, 1]] = LOW_BOUND_LAMBDA
            # model.prototype_layer.relevances[:] = model.prototype_layer.relevances[
            #                                       :] / model.prototype_layer.relevances.sum()
        # model.prototype_layer.xprotos.grad = None
        # model.prototype_layer.relevances.grad = None


        # compute the accuracy

        yspred = model.prototype_layer.yprotos[distances.argmin(axis=1)]
        acc = torch.sum(torch.eq(yspred, ytrain)).item() / float(len(xtrain))

        train_iter.set_postfix_str(
            f"Batch [{i + 1}/{len(train_loader)}, Loss: {cost.sum().item(): .3f}, Acc: {acc: .3f}"
        )

        # update the total metrics
        total_acc += acc
        total_loss += torch.sum(cost).item()

        # write a log
        if log is not None:
            log.log_values(log_loss, epoch, i + 1, torch.sum(cost).item(), acc)

    train_info['loss'] = total_loss / float(i + 1)
    train_info['train_accuracy'] = total_acc / float(i + 1)

    return train_info


def derivative_of_E_wrt_prototypes(mu_gr, dist_grad, proto_grad):
    return mu_gr.unsqueeze(-1).unsqueeze(-1) * dist_grad.unsqueeze(-1).unsqueeze(-1) * proto_grad


def derivative_of_E_wrt_relevances(mu_gr, dist_plus_grad, dist_minus_grad, rel_plus_grad, rel_minus_grad):
    return (
        mu_gr.unsqueeze(-1) * dist_plus_grad.unsqueeze(-1) * rel_plus_grad +
        mu_gr.unsqueeze(-1) * dist_minus_grad.unsqueeze(-1) * rel_minus_grad
    )
