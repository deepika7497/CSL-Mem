import os
import multiprocessing
import sys
sys.path.append(os.getcwd())

def main():
    import argparse
    import torch
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    from utils.str2bool import str2bool
    from utils.inference import inference
    from utils.load_dataset import load_dataset, create_dataloaders
    from models.resnet_k import resnet18_k
    import random
    import numpy as np
    import logging
    from tqdm import tqdm
    from minio_obj_storage import upload_numpy_as_blob, save_to_cloud

    parser = argparse.ArgumentParser(description="Train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Curvature cal parameters
    parser.add_argument("--dataset", default="cifar100", type=str, help="Set dataset to use")
    parser.add_argument("--test", action="store_true", help="Calculate curvature on Test Set")
    parser.add_argument("--lr", default=0.1, type=float, help="Learning Rate")
    parser.add_argument("--test_accuracy_display", default=True, type=str2bool, help="Test after each epoch")
    parser.add_argument("--resume", default=False, type=str2bool, help="Resume training from a saved checkpoint")
    parser.add_argument("--momentum", "--m", default=0.9, type=float, help="Momentum")
    parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, metavar="W", help="Weight decay (default: 1e-4)")
    parser.add_argument("--dedup", default=False, type=str2bool, help="Use de duplication")
    parser.add_argument("--h", default=1e-4, type=float, help="h for curvature calculation")

    # Dataloader args
    parser.add_argument("--epochs", default=200, type=int, help="Train batch size")
    parser.add_argument("--train_batch_size", default=512, type=int, help="Train batch size")
    parser.add_argument("--test_batch_size", default=512, type=int, help="Test batch size")
    parser.add_argument("--val_split", default=0.00, type=float, help="Fraction of training dataset split as validation")
    parser.add_argument("--augment", default=False, type=str2bool, help="Random horizontal flip and random crop")
    parser.add_argument("--padding_crop", default=4, type=int, help="Padding for random crop")
    parser.add_argument("--shuffle", default=False, type=str2bool, help="Shuffle the training dataset")
    parser.add_argument("--random_seed", default=0, type=int, help="Initializing the seed for reproducibility")
    parser.add_argument("--root_path", default="", type=str, help="Where to load the dataset from")
    parser.add_argument("--part", default=0, type=int, help="part = 0 or 1 for first and second part of the dataset")
    parser.add_argument("--gpu_id", default=0, type=int, help="Absolute GPU ID given by multirunner")

    # Model parameters
    parser.add_argument("--save_seed", default=False, type=str2bool, help="Save the seed")
    parser.add_argument("--use_seed", default=True, type=str2bool, help="For Random initialization")
    parser.add_argument("--suffix", default="wd1", type=str, help="Appended to model name")
    parser.add_argument("--parallel", action="store_true", help="Device in  parallel")
    parser.add_argument("--dist", action="store_true", help="Use distributed computing")
    parser.add_argument("--model_save_dir", default="./pretrained/", type=str, help="Where to load the model")
    parser.add_argument("--load_from_azure_blob", action='store_true', help="Load pre trained model from azure blob storage")
    parser.add_argument("--exp_idx", default=37, type=int, help="Model Experiment Index")
    parser.add_argument('--label_noise', default=0.01, type=float, help='Amount of label noise to add')

    global args
    args = parser.parse_args()
    args.arch = "resnet18"

    # Reproducibility settings
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # os.environ['OMP_NUM_THREADS'] = '4'

    # torch.manual_seed(args.random_seed)
    # torch.cuda.manual_seed(args.random_seed)
    # np.random.seed(args.random_seed)
    # random.seed(args.random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # try:
    #     version_list = list(map(float, torch.__version__.split(".")))
    #     if  version_list[0] <= 1 and version_list[1] < 8: ## pytorch 1.8.0 or below
    #         torch.set_deterministic(True)
    #     else:
    #         torch.use_deterministic_algorithms(True)
    # except:
    #     torch.use_deterministic_algorithms(True)

    # Setup right device to run on
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger(f"Compute Logger")
    logger.setLevel(logging.INFO)

    if not args.load_from_azure_blob:
        model_name = f"{args.dataset.lower()}_{args.arch}_{args.suffix}"
    else:
        model_name = f"{args.dataset.lower()}_{args.exp_idx}"

    handler = logging.FileHandler(os.path.join("./logs", f"score_resnet18_part_{args.part}_noisy_idx_{args.random_seed}_noise_{args.label_noise}.log"))
    formatter = logging.Formatter(fmt=f"%(asctime)s %(levelname)-8s %(message)s ", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(args)

    index = torch.load("./index/data_index_cifar100.pt")
    index = np.array(list(range(len(index))))
    if args.part == 0:
        index1 = index[:len(index) // 2]
        index2 = index[len(index) // 2:]
    else:
        index2 = index[:len(index) // 2]
        index1 = index[len(index) // 2:]        
    
    dataset_len = len(index)

    # Use the following transform for training and testing
    dataset = load_dataset(
        logger=logger,
        dataset=args.dataset,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        val_split=args.val_split,
        augment=args.augment,
        padding_crop=args.padding_crop,
        shuffle=False,
        index=index,
        root_path=args.root_path,
        random_seed=args.random_seed,
        label_noise=args.label_noise
    )

    net = resnet18_k(dataset.num_classes)
    if args.parallel:
        net = torch.nn.DataParallel(net, device_ids=[0, 1, 2])

    net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    for train_idx, train_subset_idx in enumerate([index1, index2]):
        correct = torch.zeros((2 * args.epochs, dataset_len), dtype=torch.bool)
        conf = torch.zeros_like(correct, dtype=torch.float32)

        # Define data loaders for current fold
        train_subset = torch.utils.data.Subset(dataset.train_loader.dataset.dataset, train_subset_idx)
        fold_dataset = create_dataloaders(
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
            dataset.val_loader.dataset.dataset, 
            dataset.test_loader.dataset, 
            len(train_subset),
            dataset.noisy_idxs,
            dataset.noisy_labels)

        optimizer = torch.optim.SGD(net.parameters(),
                                lr=0.01,
                                momentum=0.9,
                                weight_decay=5e-4)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(0.6*args.epochs), int(0.8*args.epochs)],
            gamma=0.1)


        for epoch in range(args.epochs):
            total = 0
            net.train()
            for inputs, targets in tqdm(fold_dataset.train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
                out = net(inputs)
                loss = criterion(out, targets)
                loss.backward()
                optimizer.step()

            state_dict = net.module.state_dict() if args.parallel else net.state_dict()
            # save_to_cloud_loc(
            #     state_dict, 
            #     'learning-dynamics-models', 
            #     f"{args.dataset.lower()}/ssft_fold_{train_idx}_resnet18_noisy_idx_{args.random_seed}_epoch_{epoch}_noise_{args.label_noise}.ckpt"
            # )
            
            with torch.no_grad():
                net.eval()
                for inputs, targets in tqdm(dataset.train_loader):
                    start_idx = total
                    stop_idx = total + len(targets)
                    idxs = index[start_idx:stop_idx]
                    total = stop_idx
                    inputs, targets = inputs.cuda(), targets.cuda()
                    out = net(inputs)
                    correct[epoch + train_idx * args.epochs][idxs] = (torch.argmax(out, 1) == targets).cpu()
                    conf[epoch + train_idx * args.epochs][idxs] = torch.nn.functional.softmax(out, 1)[range(inputs.shape[0]), targets].cpu()

            logger.info(f"Epoch: {epoch} {loss}")
            scheduler.step()

    bucket_name = "learning-dynamics-scores"
    container_dir = args.dataset.lower()
    lt_correct_file_name = f"ssft_pred_resnet18_part_{args.part}_noisy_idx_{args.random_seed}_noise_{args.label_noise}.npy"
    lt_conf_file_name = f"ssft_conf_resnet18_part_{args.part}_noisy_idx_{args.random_seed}_noise_{args.label_noise}.npy"
    upload_numpy_as_blob(bucket_name, container_dir, lt_correct_file_name, correct.numpy(), True)
    upload_numpy_as_blob(bucket_name, container_dir, lt_conf_file_name, conf.numpy(), True)
            

    if args.dist:
        destroy_process_group()


if __name__ == "__main__":
    if os.name == "nt":
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()

    main()