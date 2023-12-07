import cv2
import imageio
import pytorch_lightning as pl
import numpy as np
import torch
import PIL
import os
from skimage.io import imread
import webdataset as wds
import PIL.Image as Image
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import random

import albumentations as A

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from ldm.base_utils import read_pickle, pose_inverse
import torchvision.transforms as transforms
import torchvision
from einops import rearrange

from ldm.util import prepare_inputs


def modify_filename(full_path):
    # 分离目录和文件名
    directory, filename = os.path.split(full_path)

    try:
        # 分离文件名和扩展名
        base_name, extension = os.path.splitext(filename)

        # 提取文件名中的数字并减去6
        number = int(base_name)
        new_number = max(0, number - 8)

        # 格式化新的文件名
        new_filename = f"{new_number:03d}{extension}"

        # 构建新的完整路径
        new_full_path = os.path.join(directory, new_filename)

        return new_full_path
    except ValueError:
        # 如果文件名中的数字无法转换为整数，则返回原始文件路径
        return full_path


class SyncDreamerTrainData(Dataset):
    def __init__(self, target_dir, input_dir, uid_set_pkl, image_size=256):
        self.default_image_size = 256
        self.image_size = image_size
        self.target_dir = Path(target_dir)
        self.input_dir = Path(input_dir)

        self.uids = read_pickle(uid_set_pkl)
        print(self.uids)
        print('============= length of dataset %d =============' % len(self.uids))

        image_transforms = []
        image_transforms.extend(
            [transforms.ToTensor(), transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)
        self.num_images = 16

    def __len__(self):
        return len(self.uids)

    def load_im(self, path):
        img = imread(path)
        img = img.astype(np.float32) / 255.0
        num_channels = img.shape[2] if len(img.shape) == 3 else 1
        # print(num_channels)
        print(num_channels)
        if num_channels == 4:
            mask = img[:, :, 3:]
            img[:, :, :3] = img[:, :, :3] * mask + 1 - mask  # white background
            img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
            # img = Image.fromarray(np.uint8(img[:, :, :4] * 255.))

            return img, mask
        elif num_channels == 3:
            img = imread(path)
            img = img.astype(np.float32) / 255.0
            img = Image.fromarray(np.uint8(img * 255.))

            return img, None
        else:
            depth_map = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            # 将深度图转换为浮点型并进行归一化
            depth_map = depth_map.astype(np.float32) / 65535
            img = Image.fromarray(depth_map)
            # depth_map /= np.max(depth_map)
            # print(path)
            # print(modify_filename(path))
            # img = cv2.imread(
            #     modify_filename(path),
            # )  # The channels order is BGR due to OpenCV conventions.
            # # img = cv2.resize(img, scale, interpolation=cv2.INTER_LINEAR)
            # img = img.astype(np.float32) / 255
            # bgrd = np.dstack((img, depth_map))
            #
            # rgbd = cv2.cvtColor(bgrd, cv2.COLOR_BGRA2RGB)
            # # cv2.imwrite('/data/xzq/rgbd.exr', rgbd)
            # # print(bgrd.shape)
            # img = Image.fromarray(np.uint8(rgbd * 255))
            return img, None

        # img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        #
        # # 获取图像通道数
        # num_channels = img.shape[2] if len(img.shape) == 3 else 1
        # if num_channels == 3:
        #     img = imread(path)
        #     img = img.astype(np.float32) / 255.0
        #     img = Image.fromarray(np.uint8(img * 255.))
        #     # print("Image mode:", img.mode)
        #     return img, None
        # elif num_channels == 1:
        #     depth_map = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        #
        #     # 将深度图转换为浮点型并进行归一化
        #     depth_map = depth_map.astype(np.float32)
        #     depth_map /= np.max(depth_map)
        #     img = cv2.imread(
        #         modify_filename(path),
        #         )  # The channels order is BGR due to OpenCV conventions.
        #     # img = cv2.resize(img, scale, interpolation=cv2.INTER_LINEAR)
        #     img = img.astype(np.float32) / 255
        #     bgrd = np.dstack((img[:, :, :2], depth_map))
        #
        #     img = Image.fromarray(np.uint8(bgrd * 255))
        #     return img, None
        # img = imread(path)
        # img = img.astype(np.float32) / 255.0
        #
        # mask = img[:, :, 3:]
        # img[:, :, :3] = img[:, :, :3] * mask + 1 - mask  # white background
        #
        # img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))

    def augment_training_data(self, image, depth):
        H, W, C = image.shape
        basic_transform = [
            A.HorizontalFlip(),
            A.RandomCrop(256, 256),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue()
        ]
        self.basic_transform = basic_transform
        # if self.count % 4 == 0:
        #     alpha = random.random()
        #     beta = random.random()
        #     p = 0.75
        #
        #     l = int(alpha * W)
        #     w = int(max((W - alpha * W) * beta * p, 1))
        #
        #     image[:, l:l + w, 0] = depth[:, l:l + w]
        #     image[:, l:l + w, 1] = depth[:, l:l + w]
        #     image[:, l:l + w, 2] = depth[:, l:l + w]

        additional_targets = {'depth': 'mask'}
        aug = A.Compose(transforms=self.basic_transform,
                        additional_targets=additional_targets)
        augmented = aug(image=image, depth=depth)
        image = augmented['image']
        depth = augmented['depth']
        self.to_tensor = transforms.ToTensor()
        image = self.to_tensor(image)
        depth = self.to_tensor(depth).squeeze()



        return image, depth

    def process_im(self, im):
        # im = im.convert("RGB")
        im = im.resize((self.image_size, self.image_size), resample=PIL.Image.BICUBIC)
        return self.image_transforms(im)

    def deal_data(self, img_path, gt_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')
        image, depth = self.augment_training_data(image, depth)
        depth = depth / 1000.0  # convert in meters

        return image, depth

    def load_index(self, filename, index):
        # if type == "t":
        #     img, _ = self.load_im(os.path.join(filename, '%03d.exr' % index))
        # else:

        img, _ = self.load_im(os.path.join(filename, '%03d.png' % index))
        img = self.process_im(img)
        return img

    def load_index1(self, filename, index):
        # if type == "t":
        #     img, _ = self.load_im(os.path.join(filename, '%03d.exr' % index))
        # else:

        img_path = os.path.join(filename, '%03d.png' % index)
        dp_path = img_path.replace("input", "target2")
        # img = self.process_im(img)
        return img_path, dp_path

    def get_data_for_index(self, index):
        target_dir = os.path.join(self.target_dir, self.uids[index])
        input_dir = os.path.join(self.input_dir, self.uids[index])
        views = np.arange(0, self.num_images)
        start_view_index = np.random.randint(0, self.num_images)
        views = (views + start_view_index) % self.num_images

        target_images = []

        for si, target_index in enumerate(views):
            img = self.load_index(target_dir, target_index)
            target_images.append(img)

        # for i in range(len(target_images)):
        #     print(i)
        #     print(target_images[i].shape)
        target_images = torch.stack(target_images, 0)
        # input_img = self.load_index(input_dir, start_view_index)
        input_img = self.load_index(input_dir, start_view_index)
        input_img_path, depth_path = self.load_index1(input_dir, start_view_index)
        image, depth = self.deal_data(input_img_path, depth_path)
        K, azimuths, elevations, distances, cam_poses = read_pickle(os.path.join(input_dir, f'meta.pkl'))
        # print(elevations)

        # input_elevation = torch.from_numpy(elevations[start_view_index:start_view_index + 1].astype(np.float32))
        input_elevation = torch.from_numpy(elevations[start_view_index:start_view_index + 1].astype(np.float32))

        return {"target_image": target_images, "input_image": input_img, "input_elevation": input_elevation,
                'image': image, 'depth': depth}

    def __getitem__(self, index):
        data = self.get_data_for_index(index)
        return data


class SyncDreamerEvalData(Dataset):
    def __init__(self, image_dir):
        self.image_size = 256
        self.image_dir = Path(image_dir)
        self.crop_size = 20

        self.fns = []
        for fn in Path(image_dir).iterdir():
            if fn.suffix == '.png':
                self.fns.append(fn)
        print('============= length of dataset %d =============' % len(self.fns))

    def __len__(self):
        return len(self.fns)

    def get_data_for_index(self, index):
        input_img_fn = self.fns[index]
        elevation = int(Path(input_img_fn).stem.split('-')[-1])
        return prepare_inputs(input_img_fn, elevation, 200)

    def __getitem__(self, index):
        return self.get_data_for_index(index)


class SyncDreamerDataset(pl.LightningDataModule):
    def __init__(self, target_dir, input_dir, validation_dir, batch_size, uid_set_pkl, image_size=256, num_workers=4,
                 seed=0, **kwargs):
        super().__init__()
        self.target_dir = target_dir
        self.input_dir = input_dir
        self.validation_dir = validation_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.uid_set_pkl = uid_set_pkl
        self.seed = seed
        self.additional_args = kwargs
        self.image_size = image_size

    def setup(self, stage):
        if stage in ['fit']:
            self.train_dataset = SyncDreamerTrainData(self.target_dir, self.input_dir, uid_set_pkl=self.uid_set_pkl,
                                                      image_size=256)
            self.val_dataset = SyncDreamerEvalData(image_dir=self.validation_dir)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, seed=self.seed)
        return wds.WebLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                             shuffle=False, sampler=sampler)

    def val_dataloader(self):
        loader = wds.WebLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                               shuffle=False)
        return loader

    def test_dataloader(self):
        return wds.WebLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
