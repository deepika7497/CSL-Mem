import sys
import os
sys.path.append(os.getcwd())
import multiprocessing
import argparse
import torch
import random
import numpy as np
import logging
import json
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.inference import inference
from utils.load_dataset import load_dataset, create_dataloaders
from utils.averagemeter import AverageMeter
from utils.instantiate_model import instantiate_model, get_model_name
from minio_obj_storage import save_to_cloud
from sklearn.model_selection import StratifiedKFold

def train_one_batch(net, data, labels, optimizer, criterion, device):
    """
    Train the model on a single batch of data.

    Parameters:
    net (torch.nn.Module): The neural network model.
    data (torch.Tensor): Input data.
    labels (torch.Tensor): Target labels.
    optimizer (torch.optim.Optimizer): Optimizer for the model.
    criterion (torch.nn.Module): Loss function.
    device (torch.device): Device to run the model on.

    Returns:
    float: Loss value for the batch.
    int: Number of correct predictions.
    int: Total number of samples in the batch.
    """
    data, labels = data.to(device), labels.to(device)
    
    optimizer.zero_grad()
    out = net(data)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    
    correct = (out.max(-1)[1] == labels).sum().long().item()
    total = labels.size(0)
    
    return loss.item(), correct, total

def log_training_progress(logger, epoch, batch_idx, train_total, trainset_len, curr_acc, avg_loss):
    """
    Log training progress every 48 batches.

    Parameters:
    logger (logging.Logger): Logger for logging information.
    epoch (int): Current epoch.
    batch_idx (int): Current batch index.
    train_total (int): Total number of training samples processed.
    trainset_len (int): Total length of the training set.
    curr_acc (float): Current accuracy.
    avg_loss (float): Average loss.
    """
    if batch_idx % 48 == 0:
        logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, train_total, trainset_len, curr_acc, avg_loss
        ))

def setup_logger(config, model_name, global_rank):
    logger = logging.getLogger(f'Train Logger')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(config['log_dir'], f'train_conf_learning_{model_name}_node{global_rank}.log'))
    formatter = logging.Formatter(fmt=f'%(asctime)s [{global_rank}] %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def setup_environment(args, config):
    if args.dist:
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
    else:
        local_rank = 0
        global_rank = 0

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = '4'

    if args.reproducibility:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            version_list = list(map(float, torch.__version__.split(".")))
            if version_list[0] <= 1 and version_list[1] < 8:  # pytorch 1.8.0 or below
                torch.set_deterministic(True)
            else:
                torch.use_deterministic_algorithms(True)
        except:
            torch.use_deterministic_algorithms(True)

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    return local_rank, global_rank, device

def log_parameters_and_config(logger, args, config):
    logger.info("Training parameters and configuration:")
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')
    
    logger.info("Configuration:")
    for key, value in config.items():
        if "connection" not in key.lower():
            logger.info(f'{key}: {value}')

def save_state(net, optimizer, epoch, best_val_accuracy, best_val_loss, model_name, args, config, global_rank, fold):
    state = {
        'epoch': epoch + 1,
        'optimizer': optimizer.state_dict(),
        'model': net.module.state_dict() if args.parallel else net.state_dict(),
        'best_val_accuracy': best_val_accuracy,
        'best_val_loss': best_val_loss
    }

    if global_rank == 0:
        # temp_path = os.path.join(config['model_save_dir'], args.dataset.lower(), 'temp', model_name + '.temp')
        # torch.save(state, temp_path)
        save_model(net, args, config, model_name, epoch, fold)

def save_model(net, args, config, model_name, epoch, fold):
    state_dict = net.module.state_dict() if args.parallel else net.state_dict()
    save_to_cloud(state_dict, 'learning-dynamics-models', f"{args.dataset.lower()}/conf_learning_fold_{fold}_{model_name}_noisy_idx_{args.random_seed}_epoch_{epoch}_noise_{args.label_noise}.ckpt")

def load_checkpoint(args, config, model_name, net, optimizer):
    """
    Load the model checkpoint.

    Parameters:
    args (argparse.Namespace): Parsed arguments.
    config (dict): Configuration dictionary.
    model_name (str): Name of the model.
    net (torch.nn.Module): The neural network model.
    optimizer (torch.optim.Optimizer): Optimizer for the model.

    Returns:
    int: Starting epoch.
    float: Best validation accuracy.
    float: Best validation loss.
    """
    saved_training_state = torch.load(os.path.join(config['model_save_dir'], args.dataset.lower(), 'temp', model_name + '.temp'))
    start_epoch = saved_training_state['epoch']
    optimizer.load_state_dict(saved_training_state['optimizer'])
    net.load_state_dict(saved_training_state['model'])
    best_val_accuracy = saved_training_state['best_val_accuracy']
    best_val_loss = saved_training_state['best_val_loss']
    return start_epoch, best_val_accuracy, best_val_loss

def train(epoch_callback=None):
    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Training parameters
    parser.add_argument('--epochs', default=200, type=int, help='Set number of epochs')
    parser.add_argument('--dataset', default='cifar100', type=str, help='Set dataset to use')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning Rate')
    parser.add_argument('--test_accuracy_display', default=True, action='store_true', help='Test after each epoch')
    parser.add_argument('--resume', action='store_true', help='Resume training from a saved checkpoint')
    parser.add_argument('--momentum', '--m', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='Weight decay (default: 1e-4)')

    # Dataloader args
    parser.add_argument('--train_batch_size', default=128, type=int, help='Train batch size')
    parser.add_argument('--test_batch_size', default=128, type=int, help='Test batch size')
    parser.add_argument('--val_split', default=0.00, type=float, help='Fraction of training dataset split as validation')
    parser.add_argument('--augment', default=True, action='store_true', help='Random horizontal flip and random crop')
    parser.add_argument('--padding_crop', default=4, type=int, help='Padding for random crop')
    parser.add_argument('--shuffle', default=True, action='store_true', help='Shuffle the training dataset')
    parser.add_argument('--random_seed', default=0, type=int, help='Initializing the seed for reproducibility')
    parser.add_argument('--label_noise', default=0.00, type=float, help='Amount of label noise to add')
    parser.add_argument("--gpu_id", default=0, type=int, help="Absolute GPU ID given by multirunner")

    # Model parameters
    parser.add_argument('--save_seed', default=False, action='store_true', help='Save the seed')
    parser.add_argument('--use_seed', default=False, action='store_true', help='For random initialization')
    parser.add_argument('--arch', default='resnet18', type=str, help='Network architecture to use for training')
    parser.add_argument('--suffix', default='', type=str, help='Suffix to appended to model name')
    parser.add_argument('--parallel', default=False, action='store_true', help='Use data parallel')
    parser.add_argument('--dist', default=False, action='store_true', help='Enable distributed training')
    parser.add_argument('--reproducibility', default=False, action='store_true', help='Enable reproducibility settings')
    parser.add_argument('--cloud_save', default=False, action='store_true', help='Save in cloud')

    args = parser.parse_args()

    with open("./config.json", 'r') as f:
        config = json.load(f)

    local_rank, global_rank, device = setup_environment(args, config)
    model_name = get_model_name(args.dataset, args.arch, args.suffix)
    logger = setup_logger(config, model_name, global_rank)
    log_parameters_and_config(logger, args, config)

    dataset = load_dataset(
        dataset=args.dataset,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        val_split=args.val_split,
        augment=args.augment,
        padding_crop=args.padding_crop,
        shuffle=args.shuffle,
        index=None,
        root_path=config['data_dir'],
        random_seed=args.random_seed,
        logger=logger,
        label_noise=args.label_noise)

    net, _ = instantiate_model(
        dataset=dataset,
        arch=args.arch,
        suffix=args.suffix,
        load=args.resume,
        device=device,
        path=config['model_save_dir'],
        model_args={},
        logger=logger)

    if args.use_seed:
        save_file = os.path.join(config['seeds_dir'], f"{args.dataset.lower()}_{args.arch}.seed")
        if args.save_seed:
            logger.info("Saving Seed")
            torch.save(net.state_dict(), save_file)
        else:
            logger.info("Loading Seed")
            net.load_state_dict(torch.load(save_file))
    else:
        logger.info("Random Initialization")

    if args.resume:
        start_epoch, best_val_accuracy, best_val_loss = load_checkpoint(args, config, model_name, net, optimizer)
    else:
        start_epoch = 0
        best_val_accuracy = 0.0
        best_val_loss = float('inf')

    net = net.to(device)
    if args.parallel:
        net = torch.nn.DataParallel(net, device_ids=[0,1,2])
    if args.dist:
        net = DDP(net, device_ids=[0,1,2])

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*args.epochs), int(0.8*args.epochs)], gamma=0.1)
    num_folds = 3
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=args.random_seed)
    data = dataset.train_loader.dataset.dataset.dataset.data
    labels = dataset.train_loader.dataset.dataset.targets
    for fold, (train_idx, test_idx) in enumerate(kf.split(data, labels)):
        print(f'Fold {fold + 1}/{num_folds}')
        print('-' * 10)

        # Define data loaders for current fold
        train_subset = torch.utils.data.Subset(dataset.train_loader.dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset.train_loader.dataset, test_idx)

        kfold_dataset = create_dataloaders(
            args.dataset, 
            dataset.train_batch_size, 
            dataset.test_batch_size, 
            args.val_split, 
            args.augment, 
            args.padding_crop, 
            args.shuffle, 
            args.random_seed, 
            dataset.mean, 
            dataset.std, 
            dataset.transforms.train, 
            dataset.transforms.test, 
            dataset.transforms.val, 
            logger, 
            dataset.img_dim, 
            dataset.img_ch, 
            dataset.num_classes, 
            2, 
            train_subset, 
            val_subset, 
            dataset.test_loader.dataset, 
            len(train_subset),
            dataset.noisy_idxs,
            dataset.noisy_labels)

        for epoch in range(start_epoch, args.epochs):
            net.train()
            train_correct = 0.0
            train_total = 0.0
            save_ckpt = False
            losses = AverageMeter('Loss', ':.4e')
            for batch_idx, (data, labels) in enumerate(kfold_dataset.train_loader):
                loss, correct, total = train_one_batch(net, data, labels, optimizer, criterion, device)
                losses.update(loss)
                train_correct += correct
                train_total += total
                curr_acc = 100. * train_total / len(train_subset)
                log_training_progress(logger, epoch, batch_idx, train_total, len(train_subset), curr_acc, losses.avg)

            train_accuracy = float(train_correct) * 100.0 / float(train_total)
            logger.info('Train Epoch: {} Accuracy : {}/{} [ {:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, train_correct, train_total, train_accuracy, losses.avg))
            
            scheduler.step()
            save_state(net, optimizer, epoch, best_val_accuracy, best_val_loss, model_name, args, config, global_rank, fold)
            if epoch_callback:
                epoch_callback(net, logger, args)
            
            if args.val_split > 0.0: 
                val_correct, val_total, val_accuracy, val_loss = inference(
                    net=net,
                    data_loader=kfold_dataset.val_loader,
                    device=device,
                    loss=criterion)
                logger.info("Validation loss {:.4f}".format(val_loss))
                if val_loss <= best_val_loss:
                    best_val_accuracy = val_accuracy 
                    best_val_loss = val_loss
                    save_ckpt = True
            else:
                val_correct, val_total, val_accuracy = -1, -1, float('inf')
                if (epoch + 1) % 10 == 0:
                    save_ckpt = True
            
            if save_ckpt and global_rank == 0:
                if args.test_accuracy_display:
                    test_correct, test_total, test_accuracy = inference(net=net, data_loader=dataset.test_loader, device=device)
                    logger.info(
                        "\n================================================\n"
                        "Training set accuracy: {}/{} ({:.2f}%)\n"
                        "Validation set accuracy: {}/{} ({:.2f}%)\n"
                        "Test set accuracy: {}/{} ({:.2f}%)\n"
                        "================================================".format(
                            train_correct, train_total, train_accuracy, 
                            val_correct, val_total, val_accuracy, 
                            test_correct, test_total, test_accuracy))

        logger.info("End of training without reusing validation set")
        if args.val_split > 0.0:
            logger.info('Loading the best model on validation set')
            model_state = torch.load(args.model_save_dir + args.dataset.lower() + '/' + model_name + '.ckpt')
            if args.parallel:
                net.module.load_state_dict(model_state)
            else:
                net.load_state_dict(model_state)
            net = net.to(device)
            val_correct, val_total, val_accuracy = inference(net=net, data_loader=kfold_dataset.val_loader, device=device)
            logger.info('Validation set: Accuracy: {}/{} ({:.2f}%)'.format(val_correct, val_total, val_accuracy))

        test_correct, test_total, test_accuracy = inference(net=net, data_loader=dataset.test_loader, device=device)
        logger.info('Test set: Accuracy: {}/{} ({:.2f}%)'.format(test_correct, test_total, test_accuracy))
        train_correct, train_total, train_accuracy = inference(net=net, data_loader=kfold_dataset.train_loader, device=device)
        logger.info('Train set: Accuracy: {}/{} ({:.2f}%)'.format(train_correct, train_total, train_accuracy))

    if args.dist:
        destroy_process_group()

if __name__ == "__main__":
    if os.name == 'nt':
        multiprocessing.freeze_support()
    train()
