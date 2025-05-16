import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms.transforms import Resize
from utils.tinyimagenet import TinyImageNet
from utils.cifar100_duplicate import CIFAR100Duplicate
from utils.cifar10_duplicate import CIFAR10Duplicate
from utils.cifar10_duplicate_noisy import CIFAR10DuplicateNoisy
from utils.cifar100_duplicate_noisy import CIFAR100DuplicateNoisy
from utils.noise import UniformNoise, GaussianNoise
import os
import random

class Dict_To_Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_transform(
    test_transform,
    train_transform,
    val_transform,
    mean,
    std,
    augment,
    img_dim,
    padding_crop,
    resize=False):

    if(test_transform == None):
        test_transform = transforms.Compose([
                                                transforms.Resize((img_dim, img_dim)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std),
                                            ])

    if(val_transform == None):    
        val_transform = test_transform

    if(train_transform == None):
        if augment:
            transforms_list = [
                transforms.RandomCrop(img_dim, padding=padding_crop),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
            if resize:
                transforms_list = [transforms.Resize((img_dim, img_dim))] + transforms_list
                train_transform = transforms.Compose(transforms_list)
            else:
                train_transform = transforms.Compose(transforms_list)
        else:
            train_transform = test_transform

    return train_transform, val_transform, test_transform

def load_dataset(
    dataset='CIFAR10',
    train_batch_size=128,
    test_batch_size=128,
    val_split=0.0,
    augment=True,
    padding_crop=4,
    shuffle=True,
    random_seed=None,
    resize_shape=None,
    mean=None,
    std=None,
    train_transform=None,
    test_transform=None,
    val_transform=None,
    index=None,
    logger=None,
    root_path='',
    workers=None,
    label_noise=0.0):
    """
    Loads and prepares a dataset for training, validation, and testing.

    Parameters:
    dataset (str): Name of the dataset. Options include 'CIFAR10', 'CIFAR100', 'TinyImageNet', 'ImageNet', etc.
    train_batch_size (int): Batch size for the training dataset.
    test_batch_size (int): Batch size for the testing dataset.
    val_split (float): Percentage of the training data to be used as validation dataset.
    augment (bool): Flag to apply random horizontal flip and padding shift for data augmentation.
    padding_crop (int): Number of pixels to pad the image for cropping.
    shuffle (bool): Flag to shuffle the training and testing dataset.
    random_seed (int, optional): Seed for shuffling to reproduce results.
    resize_shape (tuple, optional): Shape to resize the images, if required.
    mean (list, optional): Mean values for normalization.
    std (list, optional): Standard deviation values for normalization.
    train_transform (torchvision.transforms.Compose, optional): Custom transformations for training data.
    test_transform (torchvision.transforms.Compose, optional): Custom transformations for testing data.
    val_transform (torchvision.transforms.Compose, optional): Custom transformations for validation data.
    index (list, optional): Indices for the dataset.
    logger (logging.Logger, optional): Logger for logging information.
    root_path (str): Root path for the datasets.
    workers (int, optional): Number of worker threads for data loading.

    Returns:
    dataset_obj (Dict_To_Obj): An object containing dataset loaders and relevant dataset information.

    Raises:
    ValueError: If the specified dataset is not supported.
    """

    # Load dataset
    # Use the following transform for training and testing
    if (dataset.lower() == 'mnist'):
        if(mean == None):
            mean = [0.1307]
        if(std == None):
            std = [0.3081]
        img_dim = 28
        img_ch = 1
        num_classes = 10
        num_worker = 2
        root = os.path.join(root_path, 'MNIST')

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = torchvision.datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=train_transform)

        valset = torchvision.datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=val_transform)

        testset = torchvision.datasets.MNIST(
            root=root,
            train=False,
            download=True,
            transform=test_transform)
        
    elif(dataset.lower() == 'cifar10'):
        if(mean == None):
            mean = [0.4914, 0.4822, 0.4465]
        if(std == None):
            std = [0.2023, 0.1994, 0.2010]

        img_dim = 32
        img_ch = 3
        num_classes = 10
        num_worker = 4
        root = os.path.join(root_path, 'CIFAR10')

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = torchvision.datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=train_transform)

        valset = torchvision.datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=val_transform)

        testset = torchvision.datasets.CIFAR10(
            root=root,
            train=False,
            download=True,
            transform=test_transform)
    
    elif(dataset.lower() == 'cifar10_duplicate'):
        if(mean == None):
            mean = [0.4914, 0.4822, 0.4465]
        if(std == None):
            std = [0.2023, 0.1994, 0.2010]

        img_dim = 32
        img_ch = 3
        num_classes = 10
        num_worker = 4
        root = os.path.join(root_path, 'CIFAR10')

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = CIFAR10Duplicate(
            root=root,
            train=True,
            download=True,
            transform=train_transform)

        valset = CIFAR10Duplicate(
            root=root,
            train=True,
            download=True,
            transform=val_transform)

        testset = CIFAR10Duplicate(
            root=root,
            train=False,
            download=True,
            transform=test_transform)

    elif(dataset.lower() == 'cifar10_duplicate_noisy'):
        if(mean == None):
            mean = [0.4914, 0.4822, 0.4465]
        if(std == None):
            std = [0.2023, 0.1994, 0.2010]

        img_dim = 32
        img_ch = 3
        num_classes = 10
        num_worker = 4
        root = os.path.join(root_path, 'CIFAR10')

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = CIFAR10DuplicateNoisy(
            root=root,
            train=True,
            download=True,
            transform=train_transform)

        valset = CIFAR10DuplicateNoisy(
            root=root,
            train=True,
            download=True,
            transform=val_transform)

        testset = CIFAR10DuplicateNoisy(
            root=root,
            train=False,
            download=True,
            transform=test_transform)

    # http://cs231n.stanford.edu/tiny-imagenet-200.zip
    elif(dataset.lower() == 'tinyimagenet'):
        if(mean == None):
            mean = [0.485, 0.456, 0.406]
        if(std == None):
            std = [0.229, 0.224, 0.225]
        root = os.path.join(root_path, 'TinyImageNet')
        img_dim = 64
        resize = False
        if(resize_shape == None):
            resize_shape = (32, 32)
            img_dim = 32
        else:
            img_dim = resize_shape[0]
            resize = True
        img_ch = 3
        num_classes = 200
        num_worker = 4
        
        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop,
            resize=resize)

        trainset = TinyImageNet(root=root, transform=test_transform, train=True) 
        valset = TinyImageNet(root=root, transform=test_transform, train=True) 
        testset = TinyImageNet(root=root, transform=test_transform, train=False)
  
    elif(dataset.lower() == 'svhn'):
        if(mean == None):
            mean = [0.4376821,  0.4437697,  0.47280442]
        if(std == None):
            std = [0.19803012, 0.20101562, 0.19703614]
        img_dim = 32
        img_ch = 3
        num_classes = 10
        num_worker = 2
        root = os.path.join(root_path, 'SVHN')

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = torchvision.datasets.SVHN(
            root=root,
            split='train',
            download=True,
            transform=train_transform)

        valset = torchvision.datasets.SVHN(
            root=root,
            split='train',
            download=True,
            transform=val_transform)

        testset = torchvision.datasets.SVHN(
            root=root,
            split='test',
            download=True, 
            transform=test_transform)

    elif(dataset.lower() == 'fmnist'):
        if(mean == None):
            mean = [0.2860]  # Mean for FashionMNIST
        if(std == None):
            std = [0.3530]   # Std for FashionMNIST

        img_dim = 28
        img_ch = 1
        num_classes = 10
        num_worker = 2
        root = os.path.join(root_path, 'FMNIST')

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = torchvision.datasets.FashionMNIST(
            root=root,
            train=True,
            download=True,
            transform=train_transform)

        valset = torchvision.datasets.FashionMNIST(
            root=root,
            train=True,
            download=True,
            transform=val_transform)

        testset = torchvision.datasets.FashionMNIST(
            root=root,
            train=False,
            download=True,
            transform=test_transform)

    elif(dataset.lower() == 'lsun'):
        if(mean == None):
            mean = [0.5071, 0.4699, 0.4326]
        if(std == None):
            std = [0.2485, 0.2492, 0.2673]
        img_dim = 32
        if(resize_shape == None):
            resize_shape = (img_dim, img_dim)
        img_ch = 3
        num_classes = 10
        num_worker = 0
       
        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        root = os.path.join(root_path, 'LSUN')
        trainset = torchvision.datasets.LSUN(
            root=root,
            classes='val',
            transform=train_transform)

        valset = torchvision.datasets.LSUN(
            root=root,
            classes='val',
            transform=val_transform)

        testset = torchvision.datasets.LSUN(
            root=root,
            classes='val',
            transform=test_transform)

    elif(dataset.lower() == 'places365'):
        if(mean == None):
            mean = [0.4578, 0.4413, 0.4078]
        if(std == None):
            std = [0.2435, 0.2418, 0.2622]
        
        img_dim = 32
        if(resize_shape == None):
            resize_shape = (img_dim, img_dim)
        img_ch = 3
        num_classes = 365
        num_worker = 4

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        root = os.path.join(root_path, 'Places365')
        trainset = torchvision.datasets.Places365(
            root=root,
            split='train-standard',
            transform=train_transform,
            download=False)

        valset = torchvision.datasets.Places365(
            root=root,
            split='train-standard',
            transform=val_transform,
            download=False)

        testset = torchvision.datasets.Places365(
            root=root,
            split='val',
            transform=test_transform,
            download=False)

    elif(dataset.lower() == 'cifar100'):
        if(mean == None):
            mean = [0.5071, 0.4867, 0.4408]
        if(std == None):
            std = [0.2675, 0.2565, 0.2761]

        img_dim = 32
        img_ch = 3
        num_classes = 100
        num_worker = 4
        root = os.path.join(root_path, 'CIFAR100')

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = torchvision.datasets.CIFAR100(
            root=root,
            train=True,
            download=True,
            transform=train_transform)

        valset = torchvision.datasets.CIFAR100(
            root=root,
            train=True,
            download=True,
            transform=val_transform)

        testset = torchvision.datasets.CIFAR100(
            root=root,
            train=False,
            download=True,
            transform=test_transform)  
        
    elif(dataset.lower() == 'cifar100_duplicate'):
        if(mean == None):
            mean = [0.5071, 0.4867, 0.4408]
        if(std == None):
            std = [0.2675, 0.2565, 0.2761]

        img_dim = 32
        img_ch = 3
        num_classes = 100
        num_worker = 4
        root = os.path.join(root_path, 'CIFAR100')

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = CIFAR100Duplicate(
            root=root,
            train=True,
            download=True,
            transform=train_transform)

        valset = CIFAR100Duplicate(
            root=root,
            train=True,
            download=True,
            transform=val_transform)

        testset = CIFAR100Duplicate(
            root=root,
            train=False,
            download=True,
            transform=test_transform)  

    elif(dataset.lower() == 'cifar100_duplicate_noisy'):
        if(mean == None):
            mean = [0.5071, 0.4867, 0.4408]
        if(std == None):
            std = [0.2675, 0.2565, 0.2761]

        img_dim = 32
        img_ch = 3
        num_classes = 100
        num_worker = 4
        root = os.path.join(root_path, 'CIFAR100')

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = CIFAR100DuplicateNoisy(
            root=root,
            train=True,
            download=True,
            transform=train_transform)

        valset = CIFAR100DuplicateNoisy(
            root=root,
            train=True,
            download=True,
            transform=val_transform)

        testset = CIFAR100DuplicateNoisy(
            root=root,
            train=False,
            download=True,
            transform=test_transform)

    elif(dataset.lower() == 'textures'):
        if(mean == None or std == None):
            raise ValueError("Mean and std for textures no supported yet!")
        img_dim = 32
        if(resize_shape == None):
            resize_shape = (img_dim, img_dim)
        img_ch = 3
        num_classes = 47
        num_worker = 4
        root = os.path.join(root_path, 'Textures')
                
        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = torchvision.datasets.ImageFolder(
            root=datapath + 'images',
            transform=train_transform)

        valset = torchvision.datasets.ImageFolder(
            root=datapath + 'images',
            transform=val_transform)

        testset = torchvision.datasets.ImageFolder(
            root=datapath + 'images',
            transform=test_transform)

    elif(dataset.lower() == 'u-noise'):
        img_dim = 32
        img_ch = 3
        num_classes = 1
        num_worker = 4

        if(mean == None or std == None):
            raise ValueError("Mean and std for textures no supported yet!")
                
        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = UniformNoise(size=(img_ch, img_dim, img_dim))
        valset = UniformNoise(size=(img_ch, img_dim, img_dim))
        testset = UniformNoise(size=(img_ch, img_dim, img_dim))

    elif(dataset.lower() == 'g-noise'):
        img_dim = 32
        img_ch = 3
        num_classes = 1
        num_worker = 4
                
        if(mean == None or std == None):
            raise ValueError("Mean and std for textures no supported yet!")
                
        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = GaussianNoise(size=(img_ch, img_dim, img_dim))
        valset = GaussianNoise(size=(img_ch, img_dim, img_dim))
        testset = GaussianNoise(size=(img_ch, img_dim, img_dim))

    elif(dataset.lower() == 'isun'):
        if(mean == None or std == None):
            raise ValueError("Mean and std for textures no supported yet!")

        img_dim = 32
        img_ch = 3
        num_classes = 1
        num_worker = 4
        datapath = os.path.join(root_path, 'iSUN')
                
        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = torchvision.datasets.ImageFolder(
            root=datapath,
            transform=train_transform)

        valset = torchvision.datasets.ImageFolder(
            root=datapath,
            transform=val_transform)

        testset = torchvision.datasets.ImageFolder(
            root=datapath,
            transform=test_transform)

    elif(dataset.lower() == 'imagenet'):
        if(mean == None):
            mean = [0.485, 0.456, 0.406]
        if(std == None):
            std = [0.229, 0.224, 0.225]

        img_dim = 224
        img_ch = 3
        num_classes = 1000
        num_worker = 40
        datapath = os.path.join(root_path, 'imagenet2012')

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(datapath, 'train'),
            transform=train_transform)

        valset = torchvision.datasets.ImageFolder(
            root=os.path.join(datapath, 'train'),
            transform=val_transform)

        testset = torchvision.datasets.ImageFolder(
            root=os.path.join(datapath, 'val'),
            transform=test_transform)

    elif(dataset.lower() == 'imagenette'):
        if(mean == None):
            mean = [0.4625, 0.4580, 0.4295]
        if(std == None):
            std = [0.2813, 0.2774, 0.3006]

        img_dim = 128
        img_ch = 3
        num_classes = 10
        num_worker = 10
        datapath = os.path.join(root_path, 'imagenette2')
                
        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop,
            resize=True)

        trainset = torchvision.datasets.ImageFolder(
            root=datapath + 'train',
            transform=train_transform)

        valset = torchvision.datasets.ImageFolder(
            root=datapath + 'train',
            transform=val_transform)

        testset = torchvision.datasets.ImageFolder(
            root=datapath + 'val',
            transform=test_transform)

    elif dataset.lower() == 'coco_cap':
        if(mean == None):
            mean = [0, 0, 0]
        if(std == None):
            std = [1, 1, 1]

        img_dim = 256
        img_ch = 3
        num_classes = 1000
        num_worker = 8
        datapath = os.path.join(root_path, 'coco_cap')
                
        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)        

        trainset = torchvision.datasets.CocoCaptions(
            root=os.path.join(datapath,'train2014'),
            annFile=os.path.join(datapath, 'annotations', 'captions_train2014.json'),
            transform=train_transform)

        valset = torchvision.datasets.CocoCaptions(
            root=os.path.join(datapath,'val2014'),
            annFile=os.path.join(datapath, 'annotations', 'captions_val2014.json'),
            transform=val_transform)

        testset = None
    
    elif(dataset.lower() == 'celeba'):
        if(mean == None):
            mean = [0.5064, 0.4258, 0.3832]
        if(std == None):
            std = [0.3080, 0.2876, 0.2870]

        img_dim = 128
        img_ch = 3
        num_classes = 40 # Because celeb a starts class numbers from 1 instead of 0
        num_worker = 2
        root = os.path.join(root_path, 'CelebA')

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop,
            resize_shape)

        trainset = torchvision.datasets.CelebA(
            root=root,
            split='train',
            target_type='attr',
            download=False,
            transform=train_transform)

        valset = torchvision.datasets.CelebA(
            root=root,
            split='train',
            target_type='attr',
            download=False,
            transform=val_transform)

        testset = torchvision.datasets.CelebA(
            root=root,
            split='test',
            target_type='attr',
            download=False,
            transform=test_transform)

    else:
        # Right way to handle exception in python 
        # see https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python
        # Explains all the traps of using exception, does a good job!! I mean the link :)
        logger.error("Unsupported dataset")
        raise ValueError("Unsupported dataset")
    
    # Split the training dataset into training and validation sets
    logger.info('Forming the sampler for train and validation split')
    if index is None:
        num_train = len(trainset)
        ind = list(range(num_train))
    else:
        num_train = len(index)
        ind = index
    
    split = int(val_split * num_train)
    logger.info(f'Split counts 0 {split} {num_train}')

    # Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    noisy_labels = []
    noisy_indices = []
    # Add label noise to the training set
    if label_noise > 0.0:
        logger.info(f"Adding {label_noise * 100}% label noise to the dataset")
        num_noisy_labels = int(label_noise * len(trainset))
        noisy_indices = random.sample(range(len(trainset)), num_noisy_labels)

        targets = np.array(trainset.targets)  # Get the original labels
        
        for idx in noisy_indices:
            original_label = targets[idx]
            possible_labels = list(range(num_classes))
            possible_labels.remove(original_label)
            noisy_label = random.choice(possible_labels)
            noisy_labels.append(noisy_label)
            targets[idx] = noisy_label

        trainset.targets = targets.tolist()  # Assign the noisy labels back to the dataset

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(ind)

    train_idx, val_idx = ind[split:], ind[:split]
    valset = torch.utils.data.Subset(trainset, val_idx)
    trainset = torch.utils.data.Subset(trainset, train_idx)
    
    num_worker = num_worker if workers is None else workers
    dataset_obj = create_dataloaders(
        dataset, 
        train_batch_size, 
        test_batch_size, 
        val_split, 
        augment, 
        padding_crop, 
        shuffle, 
        random_seed, 
        mean, 
        std, 
        train_transform, 
        test_transform, 
        val_transform, 
        logger, 
        img_dim, 
        img_ch, 
        num_classes, 
        num_worker, 
        trainset, 
        valset, 
        testset, 
        num_train,
        noisy_indices,
        noisy_labels)
    return dataset_obj

def create_dataloaders(
    dataset, 
    train_batch_size, 
    test_batch_size, 
    val_split, 
    augment, 
    padding_crop, 
    shuffle, 
    random_seed, 
    mean, 
    std, 
    train_transform, 
    test_transform, 
    val_transform, 
    logger, 
    img_dim, 
    img_ch, 
    num_classes, 
    num_worker, 
    trainset, 
    valset, 
    testset, 
    num_train,
    noisy_indices,
    noisy_labels):

    # Load dataloader
    logger.info('Loading data to the dataloader')
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=train_batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=num_worker)

    val_loader =  torch.utils.data.DataLoader(
        valset,
        batch_size=train_batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=num_worker)

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=num_worker)

    transforms_dict = {
        'train': train_transform,
        'val': val_transform,
        'test': test_transform
    }

    return_dict = {
        'name': dataset,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'num_classes': num_classes,
        'train_length': num_train,
        'testset': testset,
        'mean' : mean,
        'std': std,
        'img_dim': img_dim,
        'img_ch': img_ch,
        'train_batch_size': train_batch_size,
        'test_batch_size': test_batch_size,
        'val_split': val_split,
        'padding_crop': padding_crop,
        'augment': augment,
        'random_seed': random_seed,
        'shuffle': shuffle,
        'transforms': Dict_To_Obj(**transforms_dict),
        'num_worker': num_worker,
        'noisy_idxs': noisy_indices,
        'noisy_labels': noisy_labels
    }

    dataset_obj = Dict_To_Obj(**return_dict)
    return dataset_obj
