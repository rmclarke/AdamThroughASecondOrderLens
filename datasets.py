"""Definitions and helpers for datasets."""

import dm_pix as pix
import haiku as hk
import jax
import gzip
import pickle
import shutil
import tarfile
import transformers
import urllib.request
from PIL import Image
from pathlib import Path
from functools import partial

import numpy as np
import jax.numpy as jnp
import os
from scipy.io import loadmat

import util
from extern import text_data


def pad_dataset_for_equal_batches(dataset, batch_size=None):
    if (batch_size is None
            or len(dataset[0]) % batch_size == 0):
        return dataset

    padding_len = batch_size - len(dataset[0]) % batch_size
    return (jnp.concatenate((dataset[0],
                            jnp.full((padding_len, *dataset[0].shape[1:]),
                                    jnp.nan,
                                    dataset[0].dtype))),
            jnp.concatenate((dataset[1],
                            jnp.full((padding_len, *dataset[1].shape[1:]),
                                    jnp.nan,
                                    dataset[1].dtype))))


def make_split_datasets(name,
                        validation_proportion=0,
                        pad_to_equal_training_batches=None,
                        **dataset_kwargs):
    normalise_inputs = dataset_kwargs.pop('normalise_inputs', False)
    normalise_outputs = dataset_kwargs.pop('normalise_outputs', False)
    dataset = globals()[name]

    train_val_dataset = dataset(train=True, **dataset_kwargs)
    train_val_sizes = len(train_val_dataset) * np.array(
        [1 - validation_proportion, validation_proportion])
    train_val_sizes = np.rint(train_val_sizes).astype(int)
    # If proportional split isn't exact, may need to adjust indices to avoid
    # overflowing the dataset
    train_val_sizes[-1] -= sum(train_val_sizes) - len(train_val_dataset)
    test_dataset = dataset(train=False, **dataset_kwargs)

    if normalise_inputs:
        input_data = train_val_dataset.data
        normalise_dimension = range(input_data.ndim - 1)
        means = input_data.mean(axis=normalise_dimension)
        standard_deviations = input_data.std(axis=normalise_dimension)
        standard_deviations = jnp.where(standard_deviations == 0, 1,
                                        standard_deviations)
        normaliser = Normaliser(means, standard_deviations)
        for dataset in train_val_dataset, test_dataset:
            if dataset.transform is None:
                dataset.transform = normaliser
            else:
                original_transform = dataset.transform
                dataset.transform = lambda image: normaliser(original_transform(image))

    if normalise_outputs:
        output_data = train_val_dataset.targets
        normalise_dimension = (0, 1)
        means = output_data.mean(axis=normalise_dimension)
        standard_deviations = output_data.std(axis=normalise_dimension)
        standard_deviations = jnp.where(standard_deviations == 0, 1,
                                        standard_deviations)
        normaliser = Normaliser(means, standard_deviations)
        for dataset in train_val_dataset, test_dataset:
            if dataset.target_transform is None:
                dataset.target_transform = normaliser
            else:
                original_target_transform = dataset.target_transform
                dataset.target_transform = lambda image: normaliser(original_target_transform(image))
            setattr(dataset, 'target_unnormaliser',
                    lambda x: (x * standard_deviations) + means)

    # TODO: Ensure reproducibility of random sampler
    if validation_proportion == 0:
        datasets = (pad_dataset_for_equal_batches(train_val_dataset[:],
                                                  pad_to_equal_training_batches),
                    None,
                    test_dataset[:])
    else:
        datasets = (pad_dataset_for_equal_batches(train_val_dataset[:train_val_sizes[0]],
                                                  pad_to_equal_training_batches),
                    train_val_dataset[train_val_sizes[0]:],
                    test_dataset[:])
    return datasets, train_val_dataset.augment_inputs


def make_batches(dataset, batch_size, rng, input_augmenter):
    rng1, rng2 = jax.random.split(rng)
    if isinstance(batch_size, tuple):
        subsequence_length, batch_size = batch_size
        num_subsequences = len(dataset[0]) // (subsequence_length * batch_size)
        start_indices = jax.random.randint(rng1,
                                           shape=(num_subsequences, batch_size),
                                           minval=0,
                                           maxval=len(dataset[0])-subsequence_length)
        permutations = (jnp.expand_dims(start_indices, (1))
                        + jnp.expand_dims(jnp.arange(subsequence_length), (0, 2)))
    else:
        permutations = jax.random.permutation(rng1, len(dataset[0]))
        steps_per_epoch = max(1, len(dataset[0]) // batch_size)
        permutations = np.split(
            permutations,
            [batch_size * batch_num for batch_num in range(1, steps_per_epoch)])
    dataset = (input_augmenter(rng2, dataset[0]), dataset[1])

    for permutation in permutations:
        yield dataset[0][permutation], dataset[1][permutation]


class Normaliser():
    """Reimplementation of the torchvision `Normalize` transform, supporting a
    broader range of data sizes.
    """

    def __init__(self, means, standard_deviations):
        self.means = means
        self.standard_deviations = standard_deviations

    def __call__(self, unnormalised_data):
        return (unnormalised_data - self.means) / self.standard_deviations


class ExternalDataset():
    has_target_data = True
    track_accuracies = False

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        self.root = root

        if download:
            self.download()
            self.process()

        self.data = jnp.load(os.path.join(root, 'data.npy'))
        if self.has_target_data:
            self.targets = jnp.load(os.path.join(root, 'targets.npy'))
        else:
            self.targets = jnp.full_like(self.data, float('nan'))

        if train:
            self.data = self.data[type(self).train_val_slice]
            self.targets = self.targets[type(self).train_val_slice]
        else:  # Test
            self.data = self.data[type(self).test_slice]
            self.targets = self.targets[type(self).test_slice]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        targets = self.targets[index]

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            targets = self.target_transform(targets)

        return data, targets

    def download(self):
        """Download the necessary dataset files from the Internet."""
        for name, url in self.download_urls.items():
            download_path = os.path.join(self.root, 'raw', name)
            Path(download_path).parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url) as response, open(download_path, 'wb') as download_file:
                shutil.copyfileobj(response, download_file)

    def process(self):
        """Process the downloaded files into `data.npy` and `targets.npy`."""
        pass

    def augment_input(self, rng, data):
        """Augment the single training input `data`."""
        return rng, data

    @partial(jax.jit, static_argnums=(0,))
    @util.except_ray_jit_only_once
    def augment_inputs(self, rng, source_data):
        """Augment the inputs in `training_dataset` as specified by each subclass."""
        self._internal_rng, augmented_data =  jax.lax.scan(self.augment_input, rng, source_data)
        return augmented_data


class NullDataset(ExternalDataset):

    def __init__(self, *_, **__):
        self.data = jnp.array([0.])
        self.targets = jnp.array([None])
        self.transform = None
        self.target_transform = None


class SVHN(ExternalDataset):
    download_urls = {'train_32x32.mat': 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
                     'test_32x32.mat': 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'}
    track_accuracies = True
    train_val_slice = slice(73257)
    test_slice = slice(-26032, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, root='./data/SVHN', **kwargs)

    def process(self):
        train_mat = loadmat(os.path.join(self.root, 'raw', 'train_32x32.mat'))
        test_mat = loadmat(os.path.join(self.root, 'raw', 'test_32x32.mat'))

        data = np.concatenate((train_mat['X'],
                               test_mat['X']), axis=3)
        # Cast labels to np.int64 so they become PyTorch longs
        targets = np.concatenate((train_mat['y'].astype(np.int64).squeeze(),
                                  test_mat['y'].astype(np.int64).squeeze()), axis=0)

        # PyTorch requires labels from 0 to 9, not 1 to 10:
        np.place(targets, targets == 10, 0)
        data = np.transpose(data, (3, 0, 1, 2))

        with open(os.path.join(self.root, 'data.npy'), 'wb') as data_file:
            jnp.save(data_file, data)
        with open(os.path.join(self.root, 'targets.npy'), 'wb') as targets_file:
            jnp.save(targets_file, targets)


class CIFAR10(ExternalDataset):
    download_urls = {'base_archive.tar.gz': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'}
    track_accuracies = True
    train_val_slice = slice(50000)
    test_slice = slice(-10000, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         root='./data/CIFAR10',
                         **kwargs)

    def augment_input(self, rng, image):
        rngs = jax.random.split(rng, 3)
        image = jnp.pad(image, ((4, 4),
                                (4, 4),
                                (0, 0)))
        image = pix.random_crop(rngs[1], image, (32, 32, 3))
        image = pix.random_flip_left_right(rngs[2], image)
        return rngs[0], image

    def process(self):
        # Logic informed by https://cs.toronto.edu/~kriz/cifar.html
        with tarfile.open(os.path.join(self.root, 'raw', 'base_archive.tar.gz'),
                          'r') as archive_file:
            archive_file.extractall(os.path.join(self.root, 'raw'))

        data, targets = [], []
        batch_files = ('data_batch_1',
                       'data_batch_2',
                       'data_batch_3',
                       'data_batch_4',
                       'data_batch_5',
                       'test_batch')
        for batch_file in batch_files:
            batch_path = os.path.join(self.root,
                                      'raw',
                                      'cifar-10-batches-py',
                                      batch_file)
            with open(batch_path, 'rb') as batch_data:
                batch_dict = pickle.load(batch_data, encoding='bytes')
            data.append(
                    batch_dict[b'data']
                    .reshape(-1, 3, 32, 32)
                    .transpose((0, 2, 3, 1)))
            targets.append(
                np.array(
                    batch_dict[b'labels']))

        data = jnp.concatenate(data, axis=0)
        targets = jnp.concatenate(targets, axis=0)
        with open(os.path.join(self.root, 'data.npy'), 'wb') as data_file:
            jnp.save(data_file, data)
        with open(os.path.join(self.root, 'targets.npy'), 'wb') as targets_file:
            jnp.save(targets_file, targets)


class ImageNet(ExternalDataset):
    download_urls = {'ILSVRC2012_img_train.tar': '<REDACTED - ADD ME>',
                     'ILSVRC2012_img_val.tar': '<REDACTED - ADD ME>',
                     'ILSVRC2012_devkit_t12': '<REDACTED - ADD ME>',
                     'imagenetv2-matched-frequency.tar.gz': 'https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz'}
    track_accuracies = True
    train_val_slice = slice(1_281_167 + 50_000)
    test_slice = slice(-10_000, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         root='./data/ImageNet',
                         **kwargs)

    def augment_input(self, rng, image):
        # Logic taken from
        # https://github.com/mlcommons/algorithmic-efficiency/blob/main/algorithmic_efficiency/workloads/imagenet_resnet/imagenet_pytorch/workload.py
        # Train:
        # RandomResizedCrop(self.center_crop_size, self.scale_ratio_range, self.aspect_ratio_range)
        #     (which finally resizes to self.center_crop_size)
        # RandomHorizontalFlip
        rngs = jax.random.split(rng, 3)
        image = random_resized_crop(rngs[1],
                                    image,
                                    area_range=(0.08, 1.0),
                                    aspect_ratio_range=(3/4, 4/3),
                                    final_shape=(224, 224))
        image = pix.random_flip_left_right(rngs[2], image)
        return rngs[0], image

    def download(self):
        pass

    def process(self):
        # extract_mapping = {'ILSVRC2012_img_train.tar': 'train',
        #                    'ILSVRC2012_img_val.tar': 'val',
        #                    'imagenetv2-matched-frequency.tar.gz': 'test',
        #                    'ILSVRC2012_devkit_t12.tar.gz': 'devkit'}
        # extract_paths = {}
        # for archive_name, extract_dir in extract_mapping.items():
        #     with tarfile.open(os.path.join(self.root, 'raw', archive_name),
        #                       'r') as archive_file:
        #         extract_paths[extract_dir] = os.path.join(self.root, 'raw', extract_dir)
        #         if extract_dir == 'train':
        #             for sub_archive in archive_file:
        #                 with tarfile.open(fileobj=archive_file.extractfile(sub_archive.name),
        #                                   mode='r') as sub_archive_file:
        #                     sub_archive_file.extractall(
        #                         os.path.join(extract_paths['train'], sub_archive.name[:-4]))
        #         else:
        #             archive_file.extractall(extract_paths[extract_dir])

        extract_paths = {
            'train': os.path.join(self.root, 'raw', 'train'),
            'val': os.path.join(self.root, 'raw', 'val'),
            'test': os.path.join(self.root, 'raw', 'test'),
            'devkit': os.path.join(self.root, 'raw', 'devkit'),
        }
        data, targets = [], []

        # Training set
        sorted_class_paths = sorted(os.scandir(extract_paths['train']),
                                    key=lambda x: x.path)
        for class_id, class_path in enumerate(sorted_class_paths):
            class_size = 0
            for image_path in os.scandir(class_path):
                class_size += 1
                with Image.open(image_path.path) as image:
                    data.append(jnp.array(image))
            targets.extend([class_id] * class_size)

        # Validation set
        sorted_images = sorted(os.scandir(extract_paths['val']),
                               key=lambda x: x.path)
        for image_path in sorted_images:
            with Image.open(image_path.path) as image:
                data.append(jnp.array(image))
        targets.extend(jnp.load(os.path.join(extract_paths['devkit'], 'data', 'ILSVRC2012_validation_ground_truth.txt')))

        # Test set
        class_paths = os.scandir(os.path.join(extract_paths['test'], 'imagenetv2-matched-frequency-format-val'))
        for class_path in class_paths:
            class_size = 0
            for image_path in os.scandir(class_path):
                class_size += 1
                with Image.open(image_path.path) as image:
                    image_data = jnp.array(image)
                    # Test 'augmentation':
                    # Resize to self.resize_size
                    # CenterCrop to self.center_crop_size
                    image_data = jax.image.resize(image_data, (256, 256), method='bilinear')
                    image_data = pix.center_crop(224, 224)
                    data.append(image_data)
            targets.extend([int(class_path.name)] * class_size)

        targets = jnp.array(targets)
        with open(os.path.join(self.root, 'data.npy'), 'wb') as data_file:
            jnp.save(data_file, data)
        with open(os.path.join(self.root, 'targets.npy'), 'wb') as targets_file:
            jnp.save(targets_file, targets)


class PennTreebank(ExternalDataset):
    has_target_data = False
    download_urls = {'base_archive.tar.gz': 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'}
    train_val_slice = slice(929589 + 73760)
    test_slice = slice(-82430, None)

    def __init__(self, *args, subsequence_length=None, root='./data/PennTreebank', **kwargs):
        self.subsequence_length = subsequence_length
        super().__init__(*args,
                         root=root,
                         **kwargs)
        self.targets = jnp.concatenate((self.data[1:], jnp.full_like(self.data[0:1], jnp.nan)))

    def organise_files(self):
        # Logic informed by https://github.com/salesforce/awd-lstm-lm/blob-master-getdata.sh
        raw_directory = os.path.join(self.root, 'raw')
        with tarfile.open(os.path.join(raw_directory, 'base_archive.tar.gz'),
                          'r') as archive_file:
            archive_file.extractall(raw_directory)

        for file in ('ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt'):
            shutil.move(os.path.join(raw_directory, 'simple-examples', 'data', file),
                        os.path.join(raw_directory, file[4:]))
        return raw_directory

    def process(self):
        # Logic informed by https://github.com/salesforce/awd-lstm-lm/blob-master-getdata.sh
        raw_directory = self.organise_files()
        corpus = text_data.Corpus(raw_directory)
        data = jnp.concatenate((corpus.train, corpus.valid, corpus.test), axis=0)
        jnp.save(os.path.join(self.root, 'data.npy'), data)
        targets = jnp.concatenate((data[1:], jnp.array([float('nan')])))
        jnp.save(os.path.join(self.root, 'targets.npy'), targets)


class PennTreebankForGPT2(PennTreebank):
    def __init__(self, *args, root='./data/PennTreebankForGPT2', **kwargs):
        self.tokeniser = transformers.AutoTokenizer.from_pretrained('gpt2')
        super().__init__(*args, root=root, **kwargs)

    def process(self):
        raw_directory = self.organise_files()
        full_dataset = ""
        for data_file in ('train.txt', 'valid.txt', 'test.txt'):
            with open(os.path.join(raw_directory, data_file), 'r') as file:
                full_dataset += file.read()
        data = self.tokeniser(full_dataset, return_tensors='jax').input_ids.squeeze()
        jnp.save(os.path.join(self.root, 'data.npy'), data)
        targets = jnp.concatenate((data[1:], jnp.array([float('nan')])))
        jnp.save(os.path.join(self.root, 'targets.npy'), targets)


class FashionMNIST(ExternalDataset):

    download_urls = {'train_data.gz': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
                     'train_targets.gz': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
                     'test_data.gz': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
                     'test_targets.gz': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'}
    track_accuracies = True
    train_val_slice = slice(60000)
    test_slice = slice(-10000, None)

    def __init__(self, *args, **kwargs):
        root = kwargs.pop('root', './data/FashionMNIST')
        super().__init__(*args, root=root, **kwargs)

    def process(self):
        # Logic borrowed from https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
        with gzip.open(os.path.join(self.root, 'raw', 'train_data.gz'),
                       'rb') as raw_path:
            train_data = np.frombuffer(raw_path.read(),
                                       dtype=np.uint8,
                                       offset=16).reshape(-1, 784)
        with gzip.open(os.path.join(self.root, 'raw', 'train_targets.gz'),
                       'rb') as raw_path:
            train_targets = np.frombuffer(raw_path.read(),
                                          dtype=np.uint8,
                                          offset=8)
        with gzip.open(os.path.join(self.root, 'raw', 'test_data.gz'),
                       'rb') as raw_path:
            test_data = np.frombuffer(raw_path.read(),
                                      dtype=np.uint8,
                                      offset=16).reshape(-1, 784)
        with gzip.open(os.path.join(self.root, 'raw', 'test_targets.gz'),
                       'rb') as raw_path:
            test_targets = np.frombuffer(raw_path.read(),
                                         dtype=np.uint8,
                                         offset=8)

        data = jnp.concatenate((train_data, test_data), axis=0)
        targets = jnp.concatenate((train_targets, test_targets), axis=0)
        data = data.astype(float)
        jnp.save(os.path.join(self.root, 'data.npy'), data)
        jnp.save(os.path.join(self.root, 'targets.npy'), targets)


class UCIDataset(ExternalDataset):
    """Extend `ExternalDataset` with UCI data processing function."""

    def process(self):
        raw = np.loadtxt(
            os.path.join(self.root, 'raw', 'data_targets.txt'),
            dtype='float32')
        permutation_indices = np.loadtxt(
            os.path.join(self.root, 'permutation_indices.txt'),
            dtype='int')

        raw_permuted = raw[permutation_indices]
        with open(os.path.join(self.root, 'data.npy'), 'wb') as data_file:
            jnp.save(data_file, raw_permuted[:, :-1])
        with open(os.path.join(self.root, 'targets.npy'), 'wb') as targets_file:
            jnp.save(targets_file, raw_permuted[:, -1:])


class UCI_Energy(UCIDataset):

    train_val_slice = slice(614 + 78)
    test_slice = slice(-76, None)
    download_urls = {'data_targets.txt': 'https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets/energy/data/data.txt'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, root='./data/UCI_Energy', **kwargs)


class Kin8nm(UCIDataset):

    train_val_slice = slice(5898 + 1475)
    test_slice = slice(-819, None)
    download_urls = {'data_targets.txt': 'https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets/kin8nm/data/data.txt'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, root='./data/Kin8nm', **kwargs)


class UCI_Protein(UCIDataset):

    train_val_slice = slice(41157)
    test_slice = slice(-4573, None)
    download_urls = {'data_targets.txt': 'https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets/protein-tertiary-structure/data/data.txt'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, root='./data/UCI_Protein', **kwargs)


def random_resized_crop(rng, image, area_range, aspect_ratio_range, final_shape, num_attempts=10):
    # Logic taken from
    # https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomResizedCrop
    original_height, original_width = image.shape[:2]
    original_area = original_height * original_width

    def generate_crop(index, *_):
        target_area = original_area * jax.random.uniform(
            hk.next_rng_key(),
            minval=area_range[0],
            maxval=area_range[1])
        # Sample in log space
        aspect_ratio = jnp.exp(
            jax.random.uniform(
                hk.next_rng_key(),
                minval=jnp.log(aspect_ratio_range[0]),
                maxval=jnp.log(aspect_ratio_range[1])))
        new_width = jnp.rint(jnp.sqrt(target_area * aspect_ratio))
        new_height = jnp.rint(jnp.sqrt(target_area / aspect_ratio))
        return index+1, new_width, new_height

    def crop_valid(width, height):
        return 0 < width <= original_width and 0 < height <= original_height

    with hk.with_rng(rng):
        _, new_width, new_height = jax.lax.while_loop(
            lambda idx, w, h: idx < num_attempts and not crop_valid(w, h),
            generate_crop,
            (0, 0, 0))

        if not crop_valid(new_width, new_height):
            original_ratio = original_width / original_height
            if original_ratio < aspect_ratio_range[0]:
                new_width = original_width
                new_height = jnp.rint(original_width / aspect_ratio_range[0])
            elif original_ratio > aspect_ratio_range[1]:
                new_height = original_height
                new_width = jnp.rint(original_height / aspect_ratio_range[1])
            else:
                new_width = original_width
                new_height = original_height

        cropped_image = pix.random_crop(hk.next_rng_key(), image, (new_height, new_width, 3))
        return jax.image.resize(cropped_image, final_shape, method='bilinear')
