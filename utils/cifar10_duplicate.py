import torch
from torchvision.datasets import CIFAR10
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from minio_obj_storage import get_numpy_from_cloud, upload_numpy_as_blob

class CIFAR10Duplicate(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, num_duplicates=250):
        # Initialize the CIFAR10 dataset
        self.dataset = CIFAR10(root=root, train=train, transform=transform, 
                                target_transform=target_transform, download=download)
        self.num_duplicates = num_duplicates
        if train:
            self.indices_map = self.generate_random_mappings()
        else:
            self.indices_map = {}
    
    def generate_random_mappings(self):
        # Generate 100 unique random pairs of indices within the dataset length
        num_samples = len(self.dataset)
        all_indices = np.arange(num_samples)
        np.random.shuffle(all_indices)
        indices_pairs = np.random.choice(all_indices, size=(self.num_duplicates, 2), replace=False)
        
        try:
            indices_map = get_numpy_from_cloud('learning-dynamics-models', 'cifar10_duplicate', 'duplicate_index_map.npy').item()
            # print(f"Found mapping {indices_map}")
        except Exception as e:
            # Initialize the hash map
            indices_map = {}
            # print(f"Could not find mapping {e.args}")
            
            # Fill the hash map with mappings from requested indices to (target index, custom label) tuples
            for idx_a, idx_b in indices_pairs:
                true_label = self.dataset.targets[idx_b]
                possible_labels = set(range(10)) - {true_label}  # CIFAR10 has 10 classes
                custom_label = np.random.choice(list(possible_labels))
                indices_map[idx_a] = (idx_b, custom_label)
            
            upload_numpy_as_blob('learning-dynamics-models', 'cifar10_duplicate', 'duplicate_index_map.npy', indices_map)

        return indices_map

    def __getitem__(self, index):
        if index in self.indices_map:
            # Retrieve the target index and custom label from the hash map
            idx_b, custom_label = self.indices_map[index]
            img, _ = self.dataset[idx_b]
            return img, custom_label
        else:
            # If the index is not part of the specified mappings, return the original data
            return self.dataset[index]

    def __len__(self):
        return len(self.dataset)