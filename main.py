"""
Modified Script - Original Source: https://github.com/M-Nauta/ProtoTree/tree/main

Description:
This script is a modified version of the original script by M. Nauta.
Modifications have been made to suit specific requirements or preferences.

"""

from util.load_data_subspace import *
from lvqmodel.model import *
from util.save import *

from util.args import get_args, save_args
from util.optimizer_sgd import get_optimizer
from torch.utils.data import DataLoader
from lvqmodel.train import train_epoch
from lvqmodel.test import eval


def run_model(args=None):
    args = args or get_args()

    # Create a logger
    log = Log(args.log_dir)
    print("Log dir: ", args.log_dir, flush=True)
    # Create a csv log for storing the test accuracy, mean train accuracy and mean loss for each epoch
    log.create_log('log_epoch_overview', 'epoch', 'test_acc', 'train_acc', 'train_loss_during_epoch')
    # Log the run arguments
    save_args(args, log.metadata_dir)

    if not args.disable_cuda and torch.cuda.is_available():
        # device = torch.device('cuda')
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')

    if torch.cuda.is_available():
         torch.set_default_device(torch.cuda.FloatTensor)

    # Log which device was actually used
    log.log_message('Device used: '+str(device))

    # Create a log for logging the loss values
    log_prefix = 'log_train_epochs'
    log_loss = log_prefix+'_losses'
    log.create_log(log_loss, 'epoch', 'batch', 'loss', 'batch_train_acc')

    # ********* load data
    fold_number = 3

    dataset_train = DataSet(fold_number=fold_number, train=True)
    dataset_test = DataSet(fold_number=fold_number, train=False)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size_train, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test, shuffle=True)

    # dataset = DataSet('./data/ETH-80/', img_size=(20, 20), dim=10)
    # splitter = split_train_test(dataset, test_rate=0.5, seed=42)
    # train_loader = DataLoader(splitter[0], batch_size=args.batch_size_train, shuffle=True)
    # test_loader = DataLoader(splitter[1], batch_size=args.batch_size_test, shuffle=True)

    for x, y in train_loader:
        img_size = x.shape[-2]
        break


    # create GRLGQ model
    model = Model(img_size=img_size, num_classes=8, args=args)
    model = model.to(device=device)
    model.save(f"{log.checkpoint_dir}/model_init")

    loss = get_loss_function(args)

    # Create optimizer
    optimizer_protos, optimizer_rel, _, _ = get_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_protos, milestones=args.milestones, gamma=args.gamma)

    best_train_acc = 0.
    best_test_acc = 0.
    for epoch in range(1, args.nepochs + 1):
        log.log_message("\nEpoch %i" % epoch)
        train_info = train_epoch(
            model,
            train_loader,
            epoch,
            loss,
            args,
            optimizer_protos,
            optimizer_rel,
            device,
            log,
            log_prefix=log_prefix,
            progress_prefix='Train Epoch'
        )

        # save model
        save_model(model, epoch, log, args)

        # TODO: complete the following
        best_train_acc = save_best_train_model(model, best_train_acc, train_info['train_accuracy'], log)

        # Evaluate model
        if args.nepochs > 100:
            if epoch % 10 == 0 or epoch == args.nepochs:
                eval_info = eval(model, test_loader, epoch, loss, device, log)
                best_test_acc = save_best_test_model(model, best_test_acc,
                                                     eval_info['test_accuracy'], log)
                log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], train_info['train_accuracy'],
                               train_info['loss'])
            else:
                log.log_values('log_epoch_overview', epoch, "n.a.", train_info['train_accuracy'], train_info['loss'])
        else:
            eval_info = eval(model, test_loader, epoch, loss, device, log)
            best_test_acc = save_best_test_model(model, best_test_acc, eval_info['test_accuracy'],
                                                 log)
            log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], train_info['train_accuracy'],
                           train_info['loss'])

        # break

        # update parameters
        scheduler.step()


    log.log_message("\nTraining Finished. Best training accuracy was %s, best test accuracy was %s\n"%(str(best_train_acc), str(best_test_acc)))


if __name__ == '__main__':
    args = get_args()
    run_model(args)