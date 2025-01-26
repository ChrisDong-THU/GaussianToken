import os
import csv
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from .utils import pil_loader, get_env


preprocess: dict = {
    'miniimagenet': transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'cifar': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
}


class MiniImagenet(data.Dataset):
    def __init__(self, split='train', transform='miniimagenet', target_transform=None):
        super(MiniImagenet, self).__init__()
        self.transform = preprocess[transform] if isinstance(transform, str) else transform
        self.target_transform = target_transform

        root = get_env('MINI_IMAGENET_ROOT')
        self.image_folder = os.path.join(os.path.expanduser(root), 'images')
        splits = {
            'train': 'train.csv',
            'val': 'val.csv',
            'test': 'test.csv'
        }
        try:
            split = splits[split]
        except:
            raise ValueError('Split "{}" not found. Available splits are: {}'.format(
                split, ', '.join(splits.keys())
            ))
        
        self.split_filename = os.path.join(os.path.expanduser(root), split)
        if not self._check_exists():
            raise RuntimeError('Dataset not found at {}'.format(root))

        # Extract filenames and labels
        self._data = []
        with open(self.split_filename, 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip the header
            for line in reader:
                self._data.append(tuple(line))
                
        # self._data = self._data[:400] # For debugging
        self._fit_label_encoding()

    def __getitem__(self, index):
        filename, label = self._data[index]
        image = pil_loader(os.path.join(self.image_folder, filename))
        label = self._label_encoder[label]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return {"image": image, "label": label}

    def _fit_label_encoding(self):
        _, labels = zip(*self._data)
        unique_labels = set(labels)
        self._label_encoder = dict((label, idx)
            for (idx, label) in enumerate(unique_labels))

    def _check_exists(self):
        return (os.path.exists(self.image_folder) 
            and os.path.exists(self.split_filename))

    def __len__(self):
        return len(self._data)


class MyCIFAR(datasets.CIFAR100):
    def __init__(self, train=True, transform='cifar', target_transform=None, download=False):
        root = get_env('CIFAR100_ROOT')
        super(MyCIFAR, self).__init__(root, train=train, transform=preprocess[transform], target_transform=target_transform, download=download)

    def __getitem__(self, index):
        image, label = super(MyCIFAR, self).__getitem__(index)
        return {"image": image, "label": label}
