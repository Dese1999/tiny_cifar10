import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from collections import defaultdict
import random

class CIFAR10:
    def __init__(self, cfg):
        # Define data directory, batch size, and number of workers from config
        data_dir = cfg.data
        batch_size = cfg.batch_size if not cfg.eval_linear else cfg.linear_batch_size
        num_workers = cfg.num_threads

        # Define transformations for training data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),       # Randomly crop images to 32x32 with padding
            transforms.RandomHorizontalFlip(),          # Randomly flip images horizontally
            transforms.ToTensor(),                      # Convert images to PyTorch tensors
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize with CIFAR-10 mean and std
        ])
        
        # Define transformations for test data
        transform_test = transforms.Compose([
            transforms.ToTensor(),                      # Convert images to PyTorch tensors
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize with CIFAR-10 mean and std
        ])

        # Load CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

        # Set sampler type based on config
        sampler_type = "pair" if cfg.cs_kd else "default"
        train_sampler = None  # Add specific implementation for 'pair' sampler if needed

        # Create DataLoaders for training and testing
        self.train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None),
                                       sampler=train_sampler, num_workers=num_workers, pin_memory=True)
        self.tst_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)
        self.val_loader = self.tst_loader  # Validation loader is same as test loader in this case
        self.num_classes = 10  # Number of classes in CIFAR-10
class CIFAR10val:
    def __init__(self, cfg):
        data_dir = cfg.data
        batch_size = cfg.batch_size
        num_workers = cfg.num_threads
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Load CIFAR-10 dataset
        full_trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        val_size = int(0.1 * len(full_trainset))  # 10% for validation
        train_size = len(full_trainset) - val_size
        train_indices, val_indices = torch.randperm(len(full_trainset)).split([train_size, val_size])
        
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        valset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_val)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_val)  # Load test set

        # Apply subsampling to training set
        if hasattr(cfg, 'target_samples_per_class_ratio') and cfg.target_samples_per_class_ratio < 1.0:
            if cfg.seed is not None:
                random.seed(cfg.seed)
                np.random.seed(cfg.seed)
            classwise_indices = defaultdict(list)
            for idx in train_indices:
                y = full_trainset.targets[idx]
                classwise_indices[y].append(idx)
            subsampled_indices = []
            for cls in classwise_indices:
                cls_indices = classwise_indices[cls]
                num_samples = max(1, int(len(cls_indices) * cfg.target_samples_per_class_ratio))
                subsampled_indices.extend(random.sample(cls_indices, num_samples))
            random.shuffle(subsampled_indices)
            train_indices = subsampled_indices
            print(f"Subsampled CIFAR10val trainset size: {len(train_indices)}")

        sampler_type = "pair" if cfg.cs_kd else "default"
        train_sampler = SubsetRandomSampler(train_indices) if sampler_type == "default" else None
        val_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False,
                                       sampler=train_sampler, num_workers=num_workers, pin_memory=True)
        self.val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False,
                                     sampler=val_sampler, num_workers=num_workers, pin_memory=True)
        self.tst_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)  # Use test set
        self.num_classes = 10