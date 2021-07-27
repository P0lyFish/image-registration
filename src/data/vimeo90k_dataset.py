from resources.const import VIMEO90K_DATASET_PATH
from resources.const import IMG_NORMALIZE_STD, IMG_NORMALIZE_MEAN
import random
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os.path as osp


class Vimeo90kDataset(Dataset):
    def __init__(self, mode, cutoff=None):
        super(Vimeo90kDataset, self).__init__()

        self.mode = mode

        with open(osp.join(VIMEO90K_DATASET_PATH, f'tri_{mode}list.txt'), 'r') as datalist:
            sequence_ids = [line.strip() for line in datalist][:-1]

        self.image_paths = [
            osp.join(VIMEO90K_DATASET_PATH, 'sequences', sequence_id, 'im1.png')
            for sequence_id in sequence_ids
        ]

        if cutoff:
            if cutoff > len(self.image_paths):
                raise ValueError('Cutoff value is larger than number of data points!')
            self.image_paths = self.image_paths[:cutoff]

        random.shuffle(self.image_paths)

        self.crop = A.augmentations.crops.transforms.RandomCrop(256, 256)

        self.spatial_transform = A.Compose(
            [
                A.augmentations.geometric.transforms.Affine(
                    translate_percent={'x': [-0.15, 0.15], 'y': [-0.15, 0.15]},
                    rotate=[-10, 10],
                    shear=[-10, 10],
                    p=0.8
                ),
            ]
        )

        self.color_transform = A.Compose(
            [
                A.augmentations.transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.2,
                    p=0.6
                ),
            ]
        )

        self.to_tensor = A.Compose(
            [
                A.Normalize(mean=IMG_NORMALIZE_MEAN, std=IMG_NORMALIZE_STD),
                ToTensorV2(),
            ]
        )

        self.img_cache = []
        if self.mode == 'val':
            for i in range(len(self)):
                self.img_cache.append(self[i])

    def __getitem__(self, idx):
        if self.mode == 'val' and len(self.img_cache) > idx:
            return self.img_cache[idx]

        image_path = self.image_paths[idx]

        target = cv2.imread(image_path)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = self.crop(image=target)['image']

        src_warped = self.color_transform(image=target)['image']
        src = self.spatial_transform(
            image=src_warped
        )['image']

        src = self.to_tensor(image=src)['image']
        src_warped = self.to_tensor(image=src_warped)['image']
        target = self.to_tensor(image=target)['image']

        return {
            'input': {
                'src': src,
                'target': target
            },
            'gt': src_warped
        }

    def __len__(self):
        return len(self.image_paths)
