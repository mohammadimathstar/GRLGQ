import argparse

import torch.optim

def get_optimizer(model, args: argparse.Namespace) -> torch.optim.Optimizer:

    params_prototypes = []
    params_relevances = []

    for name, param in model.named_parameters():
        if 'xprotos' in name:
            params_prototypes.append(param)
        elif 'rel' in name:
            params_relevances.append(param)
        else:
            print(f"There are some parameter not being prototypes and relevances with name: {name}")

    proto_param_list = [
        {'params': params_prototypes, 'lr': args.lr_protos},
    ]
    rel_param_list = [
        {'params': params_relevances, 'lr': args.lr_rel}
    ]

    return torch.optim.SGD(proto_param_list), torch.optim.SGD(rel_param_list), params_prototypes, params_relevances