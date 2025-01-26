import collections
import os, hashlib
import requests
import tarfile
import urllib
import zipfile
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from .helper_types import Annotation
# from torch._six import string_classes
string_classes=str
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from tqdm import tqdm


def unpack(path):
    if path.endswith("tar.gz"):
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(path=os.path.split(path)[0])
    elif path.endswith("tar"):
        with tarfile.open(path, "r:") as tar:
            tar.extractall(path=os.path.split(path)[0])
    elif path.endswith("zip"):
        with zipfile.ZipFile(path, "r") as f:
            f.extractall(path=os.path.split(path)[0])
    else:
        raise NotImplementedError(
            "Unknown file extension: {}".format(os.path.splitext(path)[1])
        )


def reporthook(bar):
    """tqdm progress bar for downloads."""

    def hook(b=1, bsize=1, tsize=None):
        if tsize is not None:
            bar.total = tsize
        bar.update(b * bsize - bar.n)

    return hook


def get_env(name='DATASET_ROOT'):
    env = os.getenv(name, None)
    assert env is not None, 'Please set the environment variable {}'.format(name)
    return env


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def is_prepared(root):
    return Path(root).joinpath(".ready").exists()


def mark_prepared(root):
    Path(root).joinpath(".ready").touch()


def prompt_download(file_, source, target_dir, content_dir=None):
    targetpath = os.path.join(target_dir, file_)
    while not os.path.exists(targetpath):
        if content_dir is not None and os.path.exists(
            os.path.join(target_dir, content_dir)
        ):
            break
        print(
            "Please download '{}' from '{}' to '{}'.".format(file_, source, targetpath)
        )
        if content_dir is not None:
            print(
                "Or place its content into '{}'.".format(
                    os.path.join(target_dir, content_dir)
                )
            )
        input("Press Enter when done...")
    return targetpath


def download_url(file_, url, target_dir):
    targetpath = os.path.join(target_dir, file_)
    os.makedirs(target_dir, exist_ok=True)
    with tqdm(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=file_
    ) as bar:
        urllib.request.urlretrieve(url, targetpath, reporthook=reporthook(bar))
    return targetpath


def download_urls(urls, target_dir):
    paths = dict()
    for fname, url in urls.items():
        outpath = download_url(fname, url, target_dir)
        paths[fname] = outpath
    return paths


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


class KeyNotFoundError(Exception):
    def __init__(self, cause, keys=None, visited=None):
        self.cause = cause
        self.keys = keys
        self.visited = visited
        messages = list()
        if keys is not None:
            messages.append("Key not found: {}".format(keys))
        if visited is not None:
            messages.append("Visited: {}".format(visited))
        messages.append("Cause:\n{}".format(cause))
        message = "\n".join(messages)
        super().__init__(message)


def retrieve(
    list_or_dict, key, splitval="/", default=None, expand=True, pass_success=False
):
    keys = key.split(splitval)

    success = True
    try:
        visited = []
        parent = None
        last_key = None
        for key in keys:
            if callable(list_or_dict):
                if not expand:
                    raise KeyNotFoundError(
                        ValueError(
                            "Trying to get past callable node with expand=False."
                        ),
                        keys=keys,
                        visited=visited,
                    )
                list_or_dict = list_or_dict()
                parent[last_key] = list_or_dict

            last_key = key
            parent = list_or_dict

            try:
                if isinstance(list_or_dict, dict):
                    list_or_dict = list_or_dict[key]
                else:
                    list_or_dict = list_or_dict[int(key)]
            except (KeyError, IndexError, ValueError) as e:
                raise KeyNotFoundError(e, keys=keys, visited=visited)

            visited += [key]
        # final expansion of retrieved value
        if expand and callable(list_or_dict):
            list_or_dict = list_or_dict()
            parent[last_key] = list_or_dict
    except KeyNotFoundError as e:
        if default is None:
            raise e
        else:
            list_or_dict = default
            success = False

    if not pass_success:
        return list_or_dict
    else:
        return list_or_dict, success


def quadratic_crop(x, bbox, alpha=1.0):
    """bbox is xmin, ymin, xmax, ymax"""
    im_h, im_w = x.shape[:2]
    bbox = np.array(bbox, dtype=np.float32)
    bbox = np.clip(bbox, 0, max(im_h, im_w))
    center = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    l = int(alpha * max(w, h))
    l = max(l, 2)

    required_padding = -1 * min(
        center[0] - l, center[1] - l, im_w - (center[0] + l), im_h - (center[1] + l)
    )
    required_padding = int(np.ceil(required_padding))
    if required_padding > 0:
        padding = [
            [required_padding, required_padding],
            [required_padding, required_padding],
        ]
        padding += [[0, 0]] * (len(x.shape) - 2)
        x = np.pad(x, padding, "reflect")
        center = center[0] + required_padding, center[1] + required_padding
    xmin = int(center[0] - l / 2)
    ymin = int(center[1] - l / 2)
    return np.array(x[ymin : ymin + l, xmin : xmin + l, ...])


def custom_collate(batch):
    r"""source: pytorch 1.9.0, only one modification to original code """

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).view(-1, *list(elem.size()))
            print(out.shape)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
    if isinstance(elem, collections.abc.Sequence) and isinstance(elem[0], Annotation):  # added
        return batch  # added
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
