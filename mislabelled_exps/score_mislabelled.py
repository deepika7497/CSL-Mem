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
    from utils.load_dataset import load_dataset
    from utils.instantiate_model import instantiate_model
    import random
    import numpy as np
    import logging
    from tqdm import tqdm
    from minio_obj_storage import upload_numpy_as_blob, get_model_from_minio_blob
    import torchopt
    from torchvision import transforms

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
    parser.add_argument("--random_seed", default=0, type=int, help="Initializing the seed for reproducibility")
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

    if not args.load_from_azure_blob:
        model_name = f"{args.dataset.lower()}_{args.arch}_{args.suffix}"
    else:
        model_name = f"{args.dataset.lower()}_{args.arch}_{args.exp_idx}"

    handler = logging.FileHandler(os.path.join("./logs", f"score_{args.model_name.split('/')[-1].split('.')[0]}_aug_curv_scorer.log"))
    formatter = logging.Formatter(fmt=f"%(asctime)s %(levelname)-8s %(message)s ", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(args)

    if args.dataset.lower() == 'fmnist':
        dataset_len = 60000
        index = torch.Tensor(list(range(dataset_len))).to(torch.int32)
    else:
        index = torch.load("./index/data_index_cifar100.pt")
        dataset_len = len(index)

    dataset = load_dataset(
        logger=logger,
        dataset=args.dataset,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        val_split=args.val_split,
        augment=args.augment,
        padding_crop=args.padding_crop,
        shuffle=args.shuffle,
        index=index,
        root_path=args.root_path,
        random_seed=args.random_seed,
        label_noise=args.label_noise
    )

    if args.arch == 'vit_b16':
        img_size = 224
        # Use the following transform for training and testing
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset.mean, std=dataset.std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset.mean, std=dataset.std),
        ])

        dataset = load_dataset(
            dataset=args.dataset,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            val_split=args.val_split,
            augment=args.augment,
            padding_crop=args.padding_crop,
            shuffle=args.shuffle,
            index=index,
            root_path=args.root_path,
            random_seed=args.random_seed,
            distributed=args.dist,
            train_transform=transform_train,
            test_transform=transform_test
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
    
    def get_rademacher(data, h):
        v = torch.randint_like(data, high=2).to(device)
        # Generate Rademacher random variables
        for v_i in v:
            v_i[v_i == 0] = -1

        v = h * (v + 1e-7)

        return v
    
    def get_regularized_curvature_for_batch(batch_data, batch_labels, h=1e-3, niter=10, temp=1):
        num_samples = batch_data.shape[0]
        net.eval()
        regr = torch.zeros(num_samples)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        for i_iter in range(niter):
            v = torch.randint_like(batch_data, high=2).cuda()
            # Generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            v = h * (v + 1e-7)

            batch_data.requires_grad_()
            outputs_pos = net(batch_data + v)
            outputs_orig = net(batch_data)
            loss_pos = criterion(outputs_pos / temp, batch_labels).mean()
            loss_orig = criterion(outputs_orig / temp, batch_labels)
            if i_iter == 0:
                losses = loss_orig
                prob = torch.nn.functional.softmax(outputs_orig, 1)

            loss_orig = loss_orig.mean()
            grad_diff = torch.autograd.grad((loss_pos-loss_orig), batch_data)[0]

            regr += grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1).cpu().detach()
            net.zero_grad()
            if batch_data.grad is not None:
                batch_data.grad.zero_()

        curv_estimate = regr / niter
        return curv_estimate, losses, prob
    
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

    def score_true_labels_and_save(epoch, test, logger, model_name, num_classes):
        net.eval()
        dataloader = dataset.train_loader if not test else dataset.test_loader
        transformations = [
            # (0, 0, True),
            (0, 0, False),
            # (0, 4, False),
            # (-4, 0, False),
            # (0, -4, False),
            # (-4, -4, False),
            # (4, 4, True),
            # (4, 0, True),
            # (0, 4, True),
            # (-4, 0, True),
            # (0, -4, True),
        ]

        for t_idx, (shift_x, shift_y, flip) in enumerate(transformations):
            loss_curvature = torch.zeros((dataset_len))
            loss_grads = torch.zeros_like(loss_curvature)
            losses = torch.zeros_like(loss_curvature)
            probs = torch.zeros((dataset_len, num_classes))
            total = 0
            for inputs, targets in tqdm(dataloader):
                start_idx = total
                stop_idx = total + len(targets)
                idxs = index[start_idx:stop_idx]
                total = stop_idx

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs = transform_input(inputs, shift_x, shift_y, flip)
                inputs.requires_grad = True
                net.zero_grad()

                loss_curv, loss_pos, prob = get_regularized_curvature_for_batch(inputs, targets, h=args.h, niter=10)
                loss_curvature[idxs] = loss_curv.detach().clone().cpu()
                losses[idxs] = loss_pos.detach().clone().cpu()
                probs[idxs] = prob.detach().clone().cpu()

            scores_file_name = (
                f"loss_curvature_{model_name}_h{args.h}_tid{t_idx}.pt" 
                if not test 
                else f"loss_curvature_{model_name}_h{args.h}_tid{t_idx}_test.pt"
            )

            loss_file_name = (
                f"losses_{model_name}_tid{t_idx}.pt"
                if not test
                else f"losses_{model_name}_tid{t_idx}_test.pt"
            )

            prob_file_name = (
                f"prob_{model_name}_tid{t_idx}.pt"
                if not test
                else f"prob_{model_name}_tid{t_idx}_test.pt"
            )

            logger.info(f"Saving {scores_file_name}")
            blob_container = "learning-dynamics-scores"
            container_dir = args.dataset.lower()
            
            upload_numpy_as_blob(blob_container, container_dir, scores_file_name, loss_curvature.numpy(), True)
            upload_numpy_as_blob(blob_container, container_dir, loss_file_name, losses.numpy(), True)
            upload_numpy_as_blob(blob_container, container_dir, prob_file_name, probs.numpy(), True)
        return

    if args.load_from_azure_blob:
        model_state = get_model_from_minio_blob(args.container_name, args.model_name)
    else:
        model_state = torch.load(args.model_save_dir + args.dataset.lower() + "/" + model_name + f"{epoch}.ckpt")

    if args.parallel:
        net.module.load_state_dict(model_state)
    else:
        net.load_state_dict(model_state)

    net.to(device)

    test_correct, test_total, test_accuracy = inference(net=net, data_loader=dataset.train_loader, device=device)
    logger.info(" Train set: Accuracy: {}/{} ({:.2f}%)".format(test_correct, test_total, test_accuracy))

    test_correct, test_total, test_accuracy = inference(net=net, data_loader=dataset.test_loader, device=device)
    logger.info(" Test set: Accuracy: {}/{} ({:.2f}%)".format(test_correct, test_total, test_accuracy))

    model_name = ".".join(args.model_name.split("/")[-1].split(".")[:-1])

    # Calculate curvature score
    epoch = 0
    score_true_labels_and_save(epoch, args.test, logger, model_name, dataset.num_classes)

    if args.dist:
        destroy_process_group()


if __name__ == "__main__":
    if os.name == "nt":
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()

    main()
