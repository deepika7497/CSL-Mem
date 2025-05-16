import sys
import os
sys.path.append(os.getcwd())
import multiprocessing

def main():
    import argparse
    import torch
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    from utils.str2bool import str2bool
    from utils.inference import inference
    from utils.load_dataset import load_dataset, create_dataloaders
    from utils.instantiate_model import instantiate_model
    import random
    import numpy as np
    import logging
    from tqdm import tqdm
    from azure_blob_storage import upload_numpy_as_blob, get_model_from_azure_blob_file
    import torchopt
    from torchvision import transforms
    from sklearn.model_selection import StratifiedKFold

    parser = argparse.ArgumentParser(description="Compute CIFAR100 scores", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Curvature cal parameters
    parser.add_argument("--dataset", default="cifar100", type=str, help="Set dataset to use")
    parser.add_argument("--test", action="store_true", help="Calculate curvature on Test Set")
    parser.add_argument("--lr", default=0.1, type=float, help="Learning Rate")
    parser.add_argument("--test_accuracy_display", default=True, type=str2bool, help="Test after each epoch")
    parser.add_argument("--resume", default=False, type=str2bool, help="Resume training from a saved checkpoint")
    parser.add_argument("--momentum", "--m", default=0.9, type=float, help="Momentum")
    parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, metavar="W", help="Weight decay (default: 1e-4)")
    parser.add_argument("--h", default=1e-3, type=float, help="h for curvature calculation")

    # Dataloader args
    parser.add_argument("--train_batch_size", default=256, type=int, help="Train batch size")
    parser.add_argument("--test_batch_size", default=256, type=int, help="Test batch size")
    parser.add_argument("--val_split", default=0.00, type=float, help="Fraction of training dataset split as validation")
    parser.add_argument("--augment", default=False, type=str2bool, help="Random horizontal flip and random crop")
    parser.add_argument("--padding_crop", default=4, type=int, help="Padding for random crop")
    parser.add_argument("--shuffle", default=False, type=str2bool, help="Shuffle the training dataset")
    parser.add_argument("--random_seed", default=1, type=int, help="Initializing the seed for reproducibility")
    parser.add_argument("--root_path", default="", type=str, help="Where to load the dataset from")

    # Model parameters
    parser.add_argument("--save_seed",  action="store_true", help="Save the seed")
    parser.add_argument("--use_seed", action="store_true", help="For Random initialization")
    parser.add_argument("--suffix", default="wd1", type=str, help="Appended to model name")
    parser.add_argument("--parallel", action="store_true", help="Device in  parallel")
    parser.add_argument("--dist", action="store_true", help="Use distributed computing")
    parser.add_argument("--model_save_dir", default="./pretrained/", type=str, help="Where to load the model")
    parser.add_argument("--load_from_azure_blob", action='store_true', help="Load pre trained model from azure blob storage")
    parser.add_argument("--exp_idx", default=37, type=int, help="Model Experiment Index")
    parser.add_argument("--gpu_id", default=0, type=int, help="Absolute GPU ID given by multirunner")
    parser.add_argument("--model_name", default="", type=str, help="Model Name")
    parser.add_argument("--container_name", default="", type=str, help="Azure Container Name")
    parser.add_argument('--arch', default='resnet18', type=str, help='Network architecture')
    parser.add_argument('--label_noise', default=0.01, type=float, help='Amount of label noise to add')

    global args
    args = parser.parse_args()


    # Reproducibility settings
    if args.use_seed:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['OMP_NUM_THREADS'] = '4'

        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            version_list = list(map(float, torch.__version__.split(".")))
            if  version_list[0] <= 1 and version_list[1] < 8: ## pytorch 1.8.0 or below
                torch.set_deterministic(True)
            else:
                torch.use_deterministic_algorithms(True)
        except:
            torch.use_deterministic_algorithms(True)

    # Setup right device to run on
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger(f"Compute Logger")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(os.path.join("./logs", f"conf_learning_scorer_{args.model_name.split('/')[-1].split('.')[0]}.log"))
    formatter = logging.Formatter(fmt=f"%(asctime)s %(levelname)-8s %(message)s ", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(args)

    dataset = load_dataset(
        logger=logger,
        dataset=args.dataset,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        val_split=args.val_split,
        augment=args.augment,
        padding_crop=args.padding_crop,
        shuffle=args.shuffle,
        index=None,
        root_path=args.root_path,
        random_seed=args.random_seed,
        label_noise=args.label_noise
    )

    # Instantiate model 
    net, model_name = instantiate_model(
        dataset=dataset,
        arch=args.arch,
        suffix=args.suffix,
        load=args.resume,
        torch_weights=False,
        device=device,
        model_args={},
        verbose=False)

    if args.parallel:
        net = torch.nn.DataParallel(net, device_ids=[0, 1, 2])
     
    def transform_input(inputs, shift_x, shift_y, flip=False):
        """
        Apply specified shift in pixels for both width and height.
        Optionally flip the image horizontally.

        Parameters:
        - inputs: a batch of images, tensor of shape [batch_size, channels, height, width].
        - shift_x: horizontal shift value, can be negative or positive.
        - shift_y: vertical shift value, can be negative or positive.
        - flip: a boolean indicating whether to flip the image.

        Returns:
        - Transformed inputs.
        """
        # Ensure shift values are within the expected range
        shift_x = max(min(shift_x, 4), -4)
        shift_y = max(min(shift_y, 4), -4)

        # Manually create padding based on the shift directions
        if shift_x > 0:
            pad_left, pad_right = shift_x, 0
        else:
            pad_left, pad_right = 0, -shift_x
        if shift_y > 0:
            pad_top, pad_bottom = shift_y, 0
        else:
            pad_top, pad_bottom = 0, -shift_y

        # Apply padding accordingly and crop to maintain original size
        padded = torch.nn.functional.pad(inputs, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        cropped = padded[:, :, pad_top:inputs.shape[2]+pad_top, pad_left:inputs.shape[3]+pad_left]
        
        # Apply horizontal flip if required
        if flip:
            cropped = torch.flip(cropped, dims=[3])

        return cropped

    def score_true_labels_and_save(test, logger, seed, num_classes):
        net.eval()
        dataloader = dataset.train_loader if not test else dataset.test_loader
        transformations = [
            (0, 0, False),
        ]

        num_folds = 3
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=args.random_seed)
        data = dataset.train_loader.dataset.dataset.data
        labels = dataset.train_loader.dataset.dataset.targets
        pred_probs = np.zeros((dataset.train_length, dataset.num_classes))
        targets_conf_learning = np.empty((dataset.train_length))

        for fold, (train_idx, test_idx) in enumerate(kf.split(data, labels)):
            # Define data loaders for current fold
            train_subset = torch.utils.data.Subset(dataset.train_loader.dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset.train_loader.dataset, test_idx)
            model_file = f"conf_learning_fold_{fold}_{args.dataset.lower()}_resnet18_noisy_idx_{args.random_seed}_epoch_{199}_noise_{args.label_noise}.ckpt"
            model_state_dict = get_model_from_azure_blob_file('learning-dynamics-models', f"{args.dataset.lower()}/{model_file}")
            net.load_state_dict(model_state_dict)
            net.eval()
            net.to(device)

            k_fold_dataset = create_dataloaders(
                args.dataset, 
                dataset.train_batch_size, 
                dataset.test_batch_size, 
                args.val_split, 
                args.augment, 
                args.padding_crop,
                False, 
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

            with torch.no_grad():
                for t_idx, (shift_x, shift_y, flip) in enumerate(transformations):
                    probs = None
                    labels = None
                    for inputs, targets in tqdm(k_fold_dataset.val_loader):
                        inputs, targets = inputs.cuda(), targets.cuda()
                        inputs = transform_input(inputs, shift_x, shift_y, flip)
                        out = net(inputs)
                        prob = torch.nn.functional.softmax(out, dim=1)
                        prob = prob.cpu().numpy()
                        if probs is None:
                            probs = prob
                            labels = targets.cpu().numpy()
                        else:
                            probs  = np.row_stack([probs, prob])
                            labels = np.append(labels, targets.cpu().numpy())
            
            pred_probs[test_idx] = probs
            targets_conf_learning[test_idx] = labels

        scores_file_name = (
            f"conf_learning_prob_noise_idx_{args.random_seed}_noise_{args.label_noise}.pt" 
        )

        targets_file_name = (
            f"conf_learning_labels_noise_idx_{args.random_seed}_noise_{args.label_noise}.pt" 
        )

        logger.info(f"Saving {scores_file_name}")
        blob_container = "learning-dynamics-scores"
        container_dir = args.dataset.lower()
        
        upload_numpy_as_blob(blob_container, container_dir, scores_file_name, pred_probs, True)
        upload_numpy_as_blob(blob_container, container_dir, targets_file_name, targets_conf_learning, True)
        return

    # Calculate curvature score
    score_true_labels_and_save(args.test, logger, model_name, dataset.num_classes)


if __name__ == "__main__":
    if os.name == "nt":
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()

    main()
