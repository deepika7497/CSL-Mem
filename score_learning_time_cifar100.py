import os
import multiprocessing

def main():
    import argparse
    import torch
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    from utils.str2bool import str2bool
    from utils.inference import inference
    from utils.load_dataset import load_dataset
    from models.resnet_k import resnet18_k
    import random
    import numpy as np
    import logging
    from tqdm import tqdm
    from minio_obj_storage import upload_numpy_as_blob, get_model_from_minio_blob
    import json

    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument('--epochs',                 default=300,            type=int,       help='Set number of epochs')
    parser.add_argument('--dataset',                default='cifar100',     type=str,       help='Set dataset to use')
    parser.add_argument('--test',                   default=False,          type=str2bool,  help='Calculate curvature on Test Set')
    parser.add_argument('--lr',                     default=0.1,            type=float,     help='Learning Rate')
    parser.add_argument('--test_accuracy_display',  default=True,           type=str2bool,  help='Test after each epoch')
    parser.add_argument('--resume',                 default=False,          type=str2bool,  help='Resume training from a saved checkpoint')
    parser.add_argument('--momentum', '--m',        default=0.9,            type=float,     help='Momentum')
    parser.add_argument('--weight-decay', '--wd',   default=1e-4,           type=float,     metavar='W', help='Weight decay (default: 1e-4)')
    parser.add_argument('--dedup',                  default=False,          type=str2bool,  help='Use de duplication')
    parser.add_argument('--h',                      default=1e-4,           type=float,     help='h for curvature calculation')
    parser.add_argument('--n',                      default=10,             type=int,       help='n for curvature calculation')

    # Dataloader args
    parser.add_argument('--train_batch_size',       default=512,            type=int,       help='Train batch size')
    parser.add_argument('--test_batch_size',        default=512,            type=int,       help='Test batch size')
    parser.add_argument('--val_split',              default=0.00,           type=float,     help='Fraction of training dataset split as validation')
    parser.add_argument('--augment',                default=False,          type=str2bool,  help='Random horizontal flip and random crop')
    parser.add_argument('--padding_crop',           default=4,              type=int,       help='Padding for random crop')
    parser.add_argument('--shuffle',                default=False,          type=str2bool,  help='Shuffle the training dataset')
    parser.add_argument('--random_seed',            default=0,              type=int,       help='Initializing the seed for reproducibility')

    # Model parameters
    parser.add_argument('--save_seed',              default=False,          type=str2bool,  help='Save the seed')
    parser.add_argument('--use_seed',               default=True,           type=str2bool,  help='For Random initialization')
    parser.add_argument('--suffix',                 default='',             type=str,       help='Appended to model name')
    parser.add_argument('--parallel',               default=False,          type=str2bool,  help='Device in  parallel')
    parser.add_argument('--model_save_dir',         default='./pretrained/',type=str,       help='Where to load the model')

    global args
    args = parser.parse_args()
    args.arch = 'resnet18_cifar100'

    # Reproducibility settings
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
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    logger = logging.getLogger(f'Compute Logger')
    logger.setLevel(logging.INFO)

    model_name = f'{args.dataset.lower()}_{args.arch}_{args.suffix}'
    handler = logging.FileHandler(os.path.join('./logs', f'score_{model_name}_loss_grad.log'))
    formatter = logging.Formatter(
        fmt=f'%(asctime)s %(levelname)-8s %(message)s ',
        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    logger.info(args)

    with open('./config.json', 'r') as f:
        config = json.loads(f.read())


    index = torch.load("./index/data_index_cifar100.pt")
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
        shuffle=args.shuffle,
        index=index,
        root_path=config['data_dir'],
        random_seed=args.random_seed)

    # Instantiate model 
    net = resnet18_k(num_classes=dataset.num_classes)
    if args.parallel:
        net = torch.nn.DataParallel(net, device_ids=[0,1,2])

    criterion = torch.nn.CrossEntropyLoss()

    def score_true_labels_and_save(epoch, test, logger, model_name):
        scores = torch.zeros((dataset_len))
        labels = torch.zeros_like(scores, dtype=torch.long)
        net.eval()
        total = 0
        dataloader = dataset.train_loader if not test else dataset.test_loader
        with torch.no_grad():
            for (inputs, targets) in tqdm(dataloader):
                start_idx = total
                stop_idx = total + len(targets)
                idxs = index[start_idx:stop_idx]
                total = stop_idx

                inputs, targets = inputs.cuda(), targets.cuda()
                out = net(inputs)
                pred = out.argmax(1) == targets
                scores[idxs] = pred.cpu().to(torch.float32)

        scores_file_name = f"learning_time_{model_name}.pt" if not test else f"learning_time_{model_name}_test.pt"
        logger.info(f"Saving {scores_file_name}")
        blob_container = "hiker-scores"
        container_dir = "cifar100"
        upload_numpy_as_blob(blob_container, container_dir, scores_file_name, scores.numpy(), True)
        return


    for epoch in range(0, 300, 1):
        logger.info(f'Loading model for epoch {epoch}')

        model_state = get_model_from_minio_blob(container_name='hiker-models', container_file_name=f'cifar100/cifar100_resnet18_cifar100_wd1{epoch}.ckpt')
        model_name = f"cifar100_resnet18_wd1_{epoch}"
        logger.info(f"Loaded model {epoch} {model_name}")

        if args.parallel:
            net.module.load_state_dict(model_state)
        else:
            net.load_state_dict(model_state)

        net.to(device)
        test_correct, test_total, test_accuracy = inference(net=net, data_loader=dataset.test_loader, device=device)
        logger.info(' Test set: Accuracy: {}/{} ({:.2f}%)'.format(test_correct, test_total, test_accuracy))

        # Calculate curvature score
        score_true_labels_and_save(epoch, args.test, logger, model_name)

    if args.parallel:
        destroy_process_group()

if __name__ == "__main__":
    if os.name == 'nt':
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()

    main()