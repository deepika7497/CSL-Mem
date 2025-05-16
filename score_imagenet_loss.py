"""
@author: Anonym
@copyright: Anonym
"""

import os
import multiprocessing

def main():
    import argparse
    import torch
    from libdata.indexed_tfrecords import IndexedImageDataset
    from utils.str2bool import str2bool
    from utils.inference import inference_indexed_imagenet
    from torchvision.models import resnet18
    from utils.log_util import setup_logger
    from scores import get_loss_for_batch
    from tqdm import tqdm
    import json
    from minio_obj_storage import upload_numpy_as_blob, get_model_from_minio_blob
    import tensorflow as tf
    import math

    parser = argparse.ArgumentParser(description='Score ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Dataset parameters
    parser.add_argument('--epoch',                  default=0,              type=int,       help='Epoch number')
    parser.add_argument('--dataset',                default='imagenet',     type=str,       help='Set dataset to use')

    # Dataloader args
    parser.add_argument('--train_batch_size',       default=512,            type=int,       help='Train batch size')
    parser.add_argument('--test_batch_size',        default=512,            type=int,       help='Test batch size')

    # Model parameters
    parser.add_argument('--parallel',               default=True,           type=str2bool,  help='Device in  parallel')
    parser.add_argument("--gpu_id",                 default=0,              type=int,       help="Absolute GPU ID given by multirunner")

    global args
    args = parser.parse_args()
    args.arch = 'resnet18'

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        except:
            pass

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    model_name = f'{args.dataset.lower()}_{args.arch}'
    logger = setup_logger(logfile_name=f'score_loss_all_epochs_v2_{args.epoch}.log')
    logger.info(args)
    dataset_len = 1281167
    split = 'train'

    with open('./config.json', 'r') as f:
        config = json.loads(f.read())

    dataset_path = os.path.join(config['data_dir'], args.dataset)
    dataset = IndexedImageDataset(args.dataset, data_dir=dataset_path)

    # Instantiate model 
    net = resnet18(num_classes=dataset.num_classes)
    if args.parallel:
        net = torch.nn.DataParallel(net, device_ids=[0,1,2])

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    epoch = args.epoch
    logger.info(f'Loading model for epoch {epoch}')
    losses = torch.zeros((dataset_len))
    grads = torch.zeros_like(losses)

    model_name = f"imagenet_resnet18_wd1_{epoch}"
    model_state = get_model_from_minio_blob(
        bucket_name='hiker-models',
        object_name=f"imagenet/{model_name}.ckpt")

    logger.info('Loaded model from cloud')

    if args.parallel:
        net.module.load_state_dict(model_state)
    else:
        net.load_state_dict(model_state)

    net.to(device)
    test_correct, test_total, test_accuracy = inference_indexed_imagenet(net=net, dataset=dataset, device=device)
    logger.info('Test set: Accuracy: {}/{} ({:.2f}%)'.format(test_correct, test_total, test_accuracy))
    net.eval()

    # Calculate loss
    for data in tqdm(
        dataset.iterate(split, args.train_batch_size, shuffle=False, augmentation=False),
        total=math.ceil(dataset_len / args.train_batch_size)):
        images = data['image'].numpy().transpose(0, 3, 1, 2)
        inputs = torch.from_numpy(images)
        targets = torch.from_numpy(data['label'].numpy())
        idxs = data['index'].numpy()
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs.requires_grad = True
        net.zero_grad()

        loss = get_loss_for_batch(
            net,
            criterion,
            inputs,
            targets
        )

        losses[idxs] = loss.detach().clone().cpu()
    
    blob_container = "learning-dynamics-scores"
    container_dir = args.dataset.lower()

    upload_numpy_as_blob(blob_container, container_dir, f'loss_{model_name}.npy', losses.numpy(), True)
    logger.info('Done')

if __name__ == "__main__":
    if os.name == 'nt':
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()

    main()