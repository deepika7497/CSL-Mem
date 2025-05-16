import os
import multiprocessing
import sys
sys.path.append(os.getcwd())

def main():
    import argparse
    import torch
    from utils.str2bool import str2bool
    from utils.log_util import setup_logger
    from utils.inference import inference
    from utils.load_dataset import load_dataset
    from utils.instantiate_model import instantiate_model
    from scores import get_loss_for_batch, get_regularized_curvature_for_batch
    import json
    from minio_obj_storage import upload_numpy_as_blob, get_model_from_minio_blob

    parser = argparse.ArgumentParser(description='Score Duplicates', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Dataset parameters
    parser.add_argument('--epoch', default=0, type=int, help='Set number of epochs')
    parser.add_argument('--dataset', default='cifar100_duplicate', type=str, help='Set dataset to use')
    parser.add_argument('--random_seed', default=0, type=int, help='Initializing the seed for reproducibility')

    # Dataloader args
    parser.add_argument('--train_batch_size', default=128, type=int, help='Train batch size')
    parser.add_argument('--test_batch_size', default=128, type=int, help='Test batch size')
    parser.add_argument('--val_split', default=0.00, type=float, help='Fraction of training dataset split as validation')
    parser.add_argument('--augment', default=False, action='store_true', help='Random horizontal flip and random crop')
    parser.add_argument('--padding_crop', default=4, type=int, help='Padding for random crop')
    parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle the training dataset')

    # Model parameters
    parser.add_argument('--parallel', default=False, action='store_true', help='Use data parallel')
    parser.add_argument("--gpu_id", default=0, type=int, help="Absolute GPU ID given by multirunner")

    # Score parameters
    parser.add_argument("--h", default=1e-3, type=float, help="h for curvature calculation")
    parser.add_argument("--n", default=10, type=float, help="n for curvature calculation")

    global args
    args = parser.parse_args()
    args.arch = 'resnet18'

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    model_name = f'{args.dataset.lower()}_{args.arch}_seed_{args.random_seed}'
    logger = setup_logger(logfile_name=f'duplicate_{model_name}_scores_loss_{args.epoch}.log')
    logger.info(args)
    split = 'train'
    bucket_name = "learning-dynamics-scores"

    with open('./config.json', 'r') as f:
        config = json.loads(f.read())

    index = torch.load("./index/data_index_cifar100.pt")
    dataset_len = len(index)

    dataset = load_dataset(
        dataset=args.dataset,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        val_split=args.val_split,
        augment=args.augment,
        padding_crop=args.padding_crop,
        shuffle=args.shuffle,
        index=index,
        root_path=config['data_dir'],
        random_seed=args.random_seed,
        logger=logger)

    net, _ = instantiate_model(
        dataset=dataset,
        arch=args.arch,
        device=device,
        path=config['model_save_dir'],
        model_args={},
        logger=logger)

    dataset_len = dataset.train_length

    if args.parallel:
        net = torch.nn.DataParallel(net, device_ids=[0,1,2])

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    epoch = args.epoch
    logger.info(f'Loading model for epoch {epoch}')
    losses = torch.zeros((dataset_len))
    correct = torch.zeros((dataset_len)).to(torch.bool)
    curv = torch.zeros_like(losses) 
    model_name = f"{args.dataset}_resnet18_seed_{args.random_seed}_epoch_{epoch}"
    model_state = get_model_from_minio_blob(
        bucket_name='learning-dynamics-models',
        object_name=f'{args.dataset}/{model_name}.ckpt')

    logger.info('Loaded model from cloud')

    if args.parallel:
        net.module.load_state_dict(model_state)
    else:
        net.load_state_dict(model_state)

    net.to(device)
    net.eval()
    test_correct, test_total, test_accuracy = inference(net=net, data_loader=dataset.test_loader, device=device)
    logger.info('Test set: Accuracy: {}/{} ({:.2f}%)'.format(test_correct, test_total, test_accuracy))
    total = 0
    # Calculate loss
    for batch_idx, (data, labels) in enumerate(dataset.train_loader):
        inputs, targets = data.cuda(), labels.cuda()
        inputs.requires_grad = True
        net.zero_grad()

        start_idx = total
        stop_idx = total + len(targets)
        idxs = index[start_idx:stop_idx]
        total = stop_idx

        loss = get_loss_for_batch(
            net,
            criterion,
            inputs,
            targets
        )

        losses[idxs] = loss.detach().clone().cpu()

        curv_estimate = get_regularized_curvature_for_batch(
            net,
            criterion,
            inputs,
            targets,
            h=args.h,
            niter=args.n
        )

        curv[idxs] = curv_estimate.detach().clone().cpu()

        net.zero_grad()
        with torch.no_grad():
            out = net(inputs)
            pred = out.argmax(axis=1)
            correct[idxs] = (pred == targets).clone().cpu()

    container_dir = args.dataset.lower()
    upload_numpy_as_blob(bucket_name, container_dir, f'loss_{model_name}.npy', losses.numpy(), True)
    upload_numpy_as_blob(bucket_name, container_dir, f'loss_curvature_{model_name}_h_{args.h}_n_{args.n}.npy', curv.numpy(), True)
    upload_numpy_as_blob(bucket_name, container_dir, f'correct_{model_name}.npy', correct.numpy(), True)


if __name__ == "__main__":
    if os.name == 'nt':
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()

    main()