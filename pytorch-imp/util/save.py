import torch
import os
import argparse
from grlgq.model import Model
from util.log import Log

def save_model(
        model: Model,
        epoch: int,
        log: Log,
        args: argparse.Namespace
):
    model.eval()
    # Save latest model
    model.save(f'{log.checkpoint_dir}/latest')
    model.save_state(f'{log.checkpoint_dir}/latest')

    # Save model every 10 epochs
    if epoch == args.epochs or epoch%10==0:
        model.save(f'{log.checkpoint_dir}/epoch_{epoch}')
        model.save_state(f'{log.checkpoint_dir}/epoch_{epoch}')

def save_best_train_model(
        model: Model,
        best_train_acc: float,
        train_acc: float,
        log: Log
):
    model.eval()
    if train_acc > best_train_acc:
        best_train_acc = train_acc
        model.save(f'{log.checkpoint_dir}/best_train_model')
        model.save_state(f'{log.checkpoint_dir}/best_train_model')

    return best_train_acc

def save_best_test_model(
        model: Model,
        best_test_acc: float,
        test_acc: float,
        log: Log
):
    model.eval()
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        model.save(f'{log.checkpoint_dir}/best_test_model')
        model.save_state(f'{log.checkpoint_dir}/best_test_model')
    return best_test_acc

def save_tree_description(
        model: Model,
        description: str,
        log: Log
):
    model.eval()
    # Save model with description
    model.save(f'{log.checkpoint_dir}/'+description)
    model.save_state(f'{log.checkpoint_dir}/'+description)
