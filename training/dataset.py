# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import cv2
from skimage.draw import circle, line_aa

from training.utils import get_hand_mask, get_palm_mask
import math
import pycocotools.mask as maskUtils
# import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
try:
    import pyspng
except ImportError:
    pyspng = None

import random
import pickle
import glob

MISSING_VALUE = -1
# LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
#            [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
#            [0,15], [15,17], [2,16], [5,17]]

# COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
#           [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
#           [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

kptcolors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85],[255, 0, 0]]

limbseq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    # def _get_raw_labels(self):
    #     if self._raw_labels is None:
    #         self._raw_labels = self._load_raw_labels() if self._use_labels else None
    #         if self._raw_labels is None:
    #             self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
    #         assert isinstance(self._raw_labels, np.ndarray)
    #         assert self._raw_labels.shape[0] == self._raw_shape[0]
    #         assert self._raw_labels.dtype in [np.float32, np.int64]
    #         if self._raw_labels.dtype == np.int64:
    #             assert self._raw_labels.ndim == 1
    #             assert np.all(self._raw_labels >= 0)
    #     return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_person_parts_image(self, raw_idx, image, keypoints): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_pose_heatmap(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    # def _load_raw_labels(self): # to be overridden by subclass
    #     raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        person_img = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(person_img, np.ndarray)
        assert list(person_img.shape) == self.image_shape
        assert person_img.dtype == np.uint8
        pose_heatmap, keypoints = self._load_raw_pose_heatmap(self._raw_idx[idx])
        head_img, top_img, pant_img, palm_img = self._load_person_parts_image(self._raw_idx[idx], person_img, keypoints)

        if self._xflip[idx]:
            assert person_img.ndim == 3 # CHW
            person_img = person_img[:, :, ::-1]
            pose_heatmap = pose_heatmap[:, :, ::-1]
            head_img = head_img[:, :, ::-1]
            top_img = top_img[:, :, ::-1]
            pant_img = pant_img[:, :, ::-1]
            palm_img = palm_img[:, :, ::-1]

        return person_img.copy(), pose_heatmap.copy(), head_img.copy(), top_img.copy(), pant_img.copy(), palm_img.copy()

    def get_label(self, idx):
        person_img = self._load_raw_image(self._raw_idx[idx])
        pose_heatmap, keypoints = self._load_raw_pose_heatmap(self._raw_idx[idx])
        head_img, top_img, pant_img, palm_img = self._load_person_parts_image(self._raw_idx[idx], person_img, keypoints)

        return pose_heatmap.copy(), head_img.copy(), top_img.copy(), pant_img.copy(), palm_img.copy()

    # def get_label(self, idx):
    #     label = self._get_raw_labels()[self._raw_idx[idx]]
    #     if label.dtype == np.int64:
    #         onehot = np.zeros(self.label_shape, dtype=np.float32)
    #         onehot[label] = 1
    #         label = onehot
    #     return label.copy()

    # def get_details(self, idx):
    #     d = dnnlib.EasyDict()
    #     d.raw_idx = int(self._raw_idx[idx])
    #     d.xflip = (int(self._xflip[idx]) != 0)
    #     d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
    #     return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def vis_index(self):
        return self._vis_index

    # @property
    # def label_shape(self):
    #     if self._label_shape is None:
    #         raw_labels = self._get_raw_labels()
    #         if raw_labels.dtype == np.int64:
    #             self._label_shape = [int(np.max(raw_labels)) + 1]
    #         else:
    #             self._label_shape = raw_labels.shape[1:]
    #     return list(self._label_shape)

    # @property
    # def label_dim(self):
    #     assert len(self.label_shape) == 1
    #     return self.label_shape[0]

    # @property
    # def has_labels(self):
    #     return any(x != 0 for x in self.label_shape)

    # @property
    # def has_onehot_labels(self):
    #     return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW

        return image
    
    def _load_person_parts_image(self, raw_idx, person_img, keypoints):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            image_paths = json.load(f)['image_paths']

        parsing_path = image_paths[raw_idx].replace('.jpg', '_label.png')
        if parsing_path.find("/deepfashion/") > -1:
            parsing_path = parsing_path.replace("/img_320_512_image/", "/img_320_512_parsing/")
        elif parsing_path.find("/mpv/") > -1:
            parsing_path = parsing_path.replace("/MPV_320_512_image/", "/MPV_320_512_parsing/")
        elif parsing_path.find("/Zalando_512_320/") > -1:
            parsing_path = parsing_path.replace("/image/", "/parsing/")
        elif parsing_path.find("/Zalora_512_320/") > -1:
            parsing_path = parsing_path.replace("/image/", "/parsing/")

        parsing_label = cv2.imread(parsing_path)[...,0:1]
        head_mask = (parsing_label == 2).astype(np.float32) + (parsing_label == 13).astype(np.float32)
        top_mask = (parsing_label == 5).astype(np.float32) + (parsing_label == 6).astype(np.float32) + (parsing_label == 7).astype(np.float32) + (parsing_label == 11).astype(np.float32)
        pant_mask = (parsing_label == 8).astype(np.float32) + (parsing_label == 9).astype(np.float32) + (parsing_label == 12).astype(np.float32) + (parsing_label == 18).astype(np.float32) + (parsing_label == 19).astype(np.float32)

        # palm mask from keypoint
        left_hand_keypoints = keypoints[[5,6,7],:]
        right_hand_keypoints = keypoints[[2,3,4],:]
        height, width, _ = parsing_label.shape
        left_hand_up_mask, left_hand_botton_mask = get_hand_mask(left_hand_keypoints, height, width)
        right_hand_up_mask, right_hand_botton_mask = get_hand_mask(right_hand_keypoints, height, width)
        # palm mask refined by parsing
        left_hand_mask = (parsing_label == 14)
        right_hand_mask = (parsing_label == 15)
        left_palm_mask = get_palm_mask(left_hand_mask, left_hand_up_mask, left_hand_botton_mask)
        right_palm_mask = get_palm_mask(right_hand_mask, right_hand_up_mask, right_hand_botton_mask)
        palm_mask = (left_palm_mask + right_palm_mask)

        h, w, _ = parsing_label.shape
        if h > w:
            w_x_left = (int) ((h - w) / 2)
            w_x_right = h - w - w_x_left
            head_mask = np.pad(head_mask, [(0, 0), (w_x_left, w_x_right), (0, 0)], mode='constant', constant_values=0)
            top_mask = np.pad(top_mask, [(0, 0), (w_x_left, w_x_right), (0, 0)], mode='constant', constant_values=0)
            pant_mask = np.pad(pant_mask, [(0, 0), (w_x_left, w_x_right), (0, 0)], mode='constant', constant_values=0)
            palm_mask = np.pad(palm_mask, [(0, 0), (w_x_left, w_x_right), (0, 0)], mode='constant', constant_values=0)
        elif h < w:
            w_y_top = (int) ((w - h) / 2)
            w_y_bottom = w - h - w_y_top
            head_mask = np.pad(head_mask, [(w_y_top, w_y_bottom), (0, 0), (0, 0)], mode='constant', constant_values=0)
            top_mask = np.pad(top_mask, [(w_y_top, w_y_bottom), (0, 0), (0, 0)], mode='constant', constant_values=0)
            pant_mask = np.pad(pant_mask, [(w_y_top, w_y_bottom), (0, 0), (0, 0)], mode='constant', constant_values=0)
            palm_mask = np.pad(palm_mask, [(w_y_top, w_y_bottom), (0, 0), (0, 0)], mode='constant', constant_values=0)
        
        head_mask = head_mask.transpose(2, 0, 1) # HWC => CHW
        top_mask = top_mask.transpose(2, 0, 1)   # HWC => CHW
        pant_mask = pant_mask.transpose(2, 0, 1) # HWC => CHW
        palm_mask = palm_mask.transpose(2, 0, 1) # HWC => CHW

        head_mask = head_mask > 0
        top_mask = top_mask > 0
        pant_mask = pant_mask > 0
        palm_mask = palm_mask > 0

        head_img = person_img * head_mask
        top_img = person_img * top_mask
        pant_img = person_img * pant_mask
        palm_img = person_img * palm_mask

        

        return head_img, top_img, pant_img, palm_img

    def _load_raw_pose_heatmap(self, raw_idx):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            image_paths = json.load(f)['image_paths']

        pose_path = image_paths[raw_idx].replace('.jpg', '_keypoints.json')
        if pose_path.find("/deepfashion/") > -1:
            pose_path = pose_path.replace("/img_320_512_image/", "/img_320_512_keypoints/")
        elif pose_path.find("/mpv/") > -1:
            pose_path = pose_path.replace("/MPV_320_512_image/", "/MPV_320_512_keypoints/")
        elif pose_path.find("/Zalando_512_320/") > -1:
            pose_path = pose_path.replace("/image/", "/keypoints/")
        elif pose_path.find("/Zalora_512_320/") > -1:
            pose_path = pose_path.replace("/image/", "/keypoints/")

        heatmap, keypoints = self.get_pose_heatmaps(pose_path)

        return heatmap, keypoints

    # def _load_raw_labels(self):
    #     fname = 'dataset.json'
    #     if fname not in self._all_fnames:
    #         return None
    #     with self._open_file(fname) as f:
    #         labels = json.load(f)['labels']
    #     if labels is None:
    #         return None
    #     labels = dict(labels)
    #     labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
    #     labels = np.array(labels)
    #     labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
    #     return labels

    def cords_to_map(self, cords, img_size=(512, 320), sigma=8):
        result = np.zeros(img_size + cords.shape[0:1], dtype='uint8')
        for i, point in enumerate(cords):
            if point[2] == -1:
                continue
            x_matrix, y_matrix = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
            # result[..., i] = np.exp(-((x_matrix - int(point[1])) ** 2 + (y_matrix - int(point[0])) ** 2) / (2 * sigma ** 2))
            # result[..., i] = np.exp(-((x_matrix - point[0]) ** 2 + (y_matrix - point[1]) ** 2) / (2 * sigma ** 2))
            result[..., i] = np.where(((x_matrix - point[0]) ** 2 + (y_matrix - point[1]) ** 2) < (sigma ** 2), 1, 0) # 鍗婂緞涓?鐨勫渄1�7

        # padding 鎄1�7?512
        h, w, c = result.shape # (H, W, C)
        if h > w:
            w_x_left = (int) ((h - w) / 2)
            w_x_right = h - w - w_x_left
            result = np.pad(result, [(0, 0), (w_x_left, w_x_right), (0, 0)], mode='constant', constant_values=0)
        elif h < w:
            w_y_top = (int) ((w - h) / 2)
            w_y_bottom = w - h - w_y_top
            result = np.pad(result, [(w_y_top, w_y_bottom), (0, 0), (0, 0)], mode='constant', constant_values=0)
        result = result.transpose(2, 0, 1) # HWC => CHW

        return result

    def get_pose_heatmaps(self, pose_path):
        datas = None
        with open(pose_path, 'r') as f:
            datas = json.load(f)
        keypoints = np.array(datas['people'][0]['pose_keypoints_2d']).reshape((-1,3))
        for i in range(keypoints.shape[0]):
            if keypoints[i, 0] <= 0 or keypoints[i,1] <= 0:
                keypoints[i, 2] = -1
            if keypoints[i, 2] < 0.01:
                keypoints[i, 2] = -1
        pose_heatmap = self.cords_to_map(keypoints)

        return pose_heatmap, keypoints

#----------------------------------------------------------------------------

############# Dataset for full body model ############
class UvitonDatasetFull(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir' 
            dataset_list = ['Zalando_256_192', 'Zalora_256_192', 'Deepfashion_256_192', 'MPV_256_192']

            self._image_fnames = []
            self._kpt_fnames = []
            self._parsing_fnames = []
            for dataset in dataset_list:
                txt_path = os.path.join(self._path, dataset, 'train_pairs_front_list_0508.txt')
                with open(txt_path, 'r') as f:
                    for person in f.readlines():
                        person = person.strip().split()[0]
                        self._image_fnames.append(os.path.join(dataset,'image',person))
                        self._kpt_fnames.append(os.path.join(dataset,'keypoints',person.replace('.jpg', '_keypoints.json')))
                        if dataset == 'MPV_256_192':
                            self._parsing_fnames.append(os.path.join(dataset,'parsing',person.replace('.jpg','.png')))
                        else:
                            self._parsing_fnames.append(os.path.join(dataset,'parsing',person.replace('.jpg','_label.png')))
            
            vis_dir = os.path.join(self._path,'train_img_vis')
            image_list = sorted(os.listdir(vis_dir))
            vis_index = []
            for image_name in image_list:
                zalando_path = os.path.join(self._path, 'Zalando_256_192', 'image', image_name)
                deepfashion_path = os.path.join(self._path, 'Deepfashion_256_192', 'image', 'train', image_name)
                if os.path.exists(zalando_path):
                    vis_index.append(self._image_fnames.index(os.path.join('Zalando_256_192','image', image_name)))
                elif os.path.exists(deepfashion_path):
                    vis_index.append(self._image_fnames.index(os.path.join('Deepfashion_256_192','image', 'train', image_name)))

            self._vis_index = vis_index

            random_mask_acgpn_dir = os.path.join(self._path, 'train_random_mask_acgpn')
            self._random_mask_acgpn_fnames = [os.path.join(random_mask_acgpn_dir, mask_name) for mask_name in os.listdir(random_mask_acgpn_dir)]
            self._mask_acgpn_numbers = len(self._random_mask_acgpn_fnames)
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        im_shape = list((self._load_raw_image(0))[0].shape)
        raw_shape = [len(self._image_fnames)] + [im_shape[2], im_shape[0], im_shape[1]]
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        # load images --> range [0, 255]
        fname = self._image_fnames[raw_idx]
        f = os.path.join(self._path, fname)
        self.image = np.array(PIL.Image.open(f))
        im_shape = self.image.shape
        h, w = im_shape[0], im_shape[1]
        left_padding = (h-w) // 2
        right_padding = h-w-left_padding
        image = np.pad(self.image,((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(255,255))

        # load keypoints --> range [0, 1]
        fname = self._kpt_fnames[raw_idx]
        kpt = os.path.join(self._path, fname)
        pose = self.get_joints(kpt) # self.cords_to_map(kpt, im_shape[:2])
        pose = np.pad(pose,((0,0),(left_padding,right_padding),(0,0)),'constant',constant_values=(0,0))

        # load upper_cloth and lower body
        fname = self._parsing_fnames[raw_idx]
        f = os.path.join(self._path, fname)
        self.parsing = cv2.imread(f)[...,0:1]
        parsing = np.pad(self.parsing, ((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(0,0))

        shoes_mask = (parsing==18).astype(np.uint8) + (parsing==19).astype(np.uint8)
        head_mask = (parsing==1).astype(np.uint8) + (parsing==2).astype(np.uint8) + \
                    (parsing==4).astype(np.uint8) + (parsing==13).astype(np.uint8)
        palm_mask = self.get_palm(left_padding, right_padding)
        retain_mask = shoes_mask + palm_mask + head_mask

        upper_clothes_mask = (parsing==5).astype(np.uint8) + (parsing==6).astype(np.uint8) + \
                             (parsing==7).astype(np.uint8)
        lower_clothes_mask = (parsing==9).astype(np.uint8) + (parsing==12).astype(np.uint8)

        hands_mask = (parsing==14).astype(np.uint8) + (parsing==15).astype(np.uint8)
        legs_mask = (parsing==16).astype(np.uint8) + (parsing==17).astype(np.uint8)
        neck_mask = (parsing==10).astype(np.uint8)
        gt_parsing = upper_clothes_mask * 1 + lower_clothes_mask * 2 + \
                     hands_mask * 3 + legs_mask * 4 + neck_mask * 5

        upper_clothes_image = upper_clothes_mask * image
        lower_clothes_image = lower_clothes_mask * image

        upper_clothes_mask_rgb = np.concatenate([upper_clothes_mask,upper_clothes_mask,upper_clothes_mask],axis=2)
        lower_clothes_mask_rgb = np.concatenate([lower_clothes_mask,lower_clothes_mask,lower_clothes_mask],axis=2)
        upper_clothes_mask_rgb = upper_clothes_mask_rgb * 255
        lower_clothes_mask_rgb = lower_clothes_mask_rgb * 255

        norm_img, norm_img_lower, denorm_upper_img, denorm_lower_img, M_invs, denorm_hand_masks, \
                norm_clothes_masks, norm_clothes_masks_lower = self.normalize(upper_clothes_image, lower_clothes_image, \
                upper_clothes_mask_rgb, lower_clothes_mask_rgb, 2)

        return image, pose, norm_img, norm_img_lower, denorm_upper_img, denorm_lower_img, M_invs, \
                gt_parsing, denorm_hand_masks, norm_clothes_masks, norm_clothes_masks_lower, retain_mask


    def _load_raw_labels(self):
        fname = 'dataset.json'
        if not os.path.exists(os.path.join(self._path, fname)):
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels


    def cords_to_map(self, keypoints_path, img_size, old_size=None,
                     affine_matrix=None, coeffs=None, sigma=6):
        with open(keypoints_path, 'r') as f:
            keypoints_data = json.load(f)
        if len(keypoints_data['people']) == 0:
            self.cords = np.zeros((18, 3))
        else:
            self.cords = np.array(keypoints_data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
        old_size = img_size if old_size is None else old_size
        self.cords = self.cords.astype(float)
        result = np.zeros(img_size + (self.cords.shape[0],), dtype='float32')
        for i, (x, y, score) in enumerate(self.cords):
            if score < 0.1:
                continue
            x = x / old_size[0] * img_size[0]
            y = y / old_size[1] * img_size[1]
            if affine_matrix is not None:
                point_ = np.dot(affine_matrix, np.matrix([x, y, 1]).reshape(3, 1))
                x = int(point_[0])
                y = int(point_[1])
            else:
                x = int(x)
                y = int(y)
            if coeffs is not None:
                a, b, c, d, e, f, g, h = coeffs
                x = int((a * x + b * y + c) / (g * x + h * y + 1))
                y = int((d * x + e * y + f) / (g * x + h * y + 1))
            xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
            result[..., i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

        return result
    
    ############################ get palm mask start #########################################

    def get_mask_from_kps(self, kps, img_h, img_w):
        rles = maskUtils.frPyObjects(kps, img_h, img_w)
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)[...,np.newaxis].astype(np.float32)
        mask = mask * 255.0
        return mask

    def get_rectangle_mask(self, a, b, c, d, img_h, img_w):
        x1, y1 = a + (b-d)/4,   b + (c-a)/4
        x2, y2 = a - (b-d)/4,   b - (c-a)/4

        x3, y3 = c + (b-d)/4,   d + (c-a)/4
        x4, y4 = c - (b-d)/4,   d - (c-a)/4

        kps  = [x1,y1,x2,y2]

        v0_x, v0_y = c-a,   d-b
        v1_x, v1_y = x3-x1, y3-y1
        v2_x, v2_y = x4-x1, y4-y1

        cos1 = (v0_x*v1_x+v0_y*v1_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v1_x*v1_x+v1_y*v1_y))
        cos2 = (v0_x*v2_x+v0_y*v2_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v2_x*v2_x+v2_y*v2_y))

        if cos1<cos2:
            kps.extend([x3,y3,x4,y4])
        else:
            kps.extend([x4,y4,x3,y3])

        kps = np.array(kps).reshape(1,-1).tolist()
        mask = self.get_mask_from_kps(kps, img_h=img_h, img_w=img_w)

        return mask
    
    def get_hand_mask(self, hand_keypoints):
        # shoulder, elbow, wrist    
        s_x,s_y,s_c = hand_keypoints[0]
        e_x,e_y,e_c = hand_keypoints[1]
        w_x,w_y,w_c = hand_keypoints[2]

        h, w = 256, 256
        up_mask = np.ones((256,256,1),dtype=np.float32)
        bottom_mask = np.ones((256,256,1),dtype=np.float32)
        if s_c > 0.1 and e_c > 0.1:
            up_mask = self.get_rectangle_mask(s_x, s_y, e_x, e_y, h, w)
            kernel = np.ones((25,25),np.uint8)
            up_mask = cv2.dilate(up_mask,kernel,iterations=1)
            up_mask = (up_mask > 0).astype(np.float32)[...,np.newaxis]
        if e_c > 0.1 and w_c > 0.1:
            bottom_mask = self.get_rectangle_mask(e_x, e_y, w_x, w_y, h, w)
            kernel = np.ones((18,18),np.uint8)
            bottom_mask = cv2.dilate(bottom_mask,kernel,iterations=1)
            bottom_mask = (bottom_mask > 0).astype(np.float32)[...,np.newaxis]

        return up_mask, bottom_mask

    def get_palm_mask(self, hand_mask, hand_up_mask, hand_bottom_mask):
        inter_up_mask = ((hand_mask + hand_up_mask) == 2).astype(np.float32)
        hand_mask = hand_mask - inter_up_mask
        inter_bottom_mask = ((hand_mask+hand_bottom_mask)==2).astype(np.float32)
        palm_mask = hand_mask - inter_bottom_mask

        return palm_mask

    def get_palm(self, left_padding, right_padding):
        left_hand_keypoints = self.keypoints[[5,6,7],:].copy()
        right_hand_keypoints = self.keypoints[[2,3,4],:].copy()

        left_hand_keypoints[:,0] += left_padding
        right_hand_keypoints[:,0] += left_padding
        parsing = np.pad(self.parsing, ((0,0),(left_padding,right_padding),(0,0)), 'constant', constant_values=(0,0))

        left_hand_up_mask, left_hand_botton_mask = self.get_hand_mask(left_hand_keypoints)
        right_hand_up_mask, right_hand_botton_mask = self.get_hand_mask(right_hand_keypoints)

        # mask refined by parsing
        left_hand_mask = (parsing == 14).astype(np.float32)
        right_hand_mask = (parsing == 15).astype(np.float32)
        left_palm_mask = self.get_palm_mask(left_hand_mask, left_hand_up_mask, left_hand_botton_mask)
        right_palm_mask = self.get_palm_mask(right_hand_mask, right_hand_up_mask, right_hand_botton_mask)
        palm_mask = ((left_palm_mask + right_palm_mask) > 0).astype(np.uint8)

        return palm_mask

    ############################ get palm mask end #########################################

    def draw_pose_from_cords(self, pose_joints, img_size, affine_matrix=None,
                             coeffs=None, radius=2, draw_joints=True):
        colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
        if draw_joints:
            for i, p in enumerate(limbseq):
                f, t = p[0]-1, p[1]-1
                from_missing = pose_joints[f][2] < 0.1
                to_missing = pose_joints[t][2] < 0.1
                if from_missing or to_missing:
                    continue
                if not affine_matrix is None:
                    pf = np.dot(affine_matrix, np.matrix([pose_joints[f][0], pose_joints[f][1], 1]).reshape(3, 1))
                    pt = np.dot(affine_matrix, np.matrix([pose_joints[t][0], pose_joints[t][1], 1]).reshape(3, 1))
                else:
                    pf = pose_joints[f][0], pose_joints[f][1]
                    pt = pose_joints[t][0], pose_joints[t][1]
                fx, fy = pf[1], pf[0]# max(pf[1], 0), max(pf[0], 0)
                tx, ty = pt[1], pt[0]# max(pt[1], 0), max(pt[0], 0)
                fx, fy = int(fx), int(fy)# int(min(fx, 255)), int(min(fy, 191))
                tx, ty = int(tx), int(ty)# int(min(tx, 255)), int(min(ty, 191))
                cv2.line(colors, (fy, fx), (ty, tx), kptcolors[i], 2)

        for i, joint in enumerate(pose_joints):
            if pose_joints[i][2] < 0.1:
                continue
            if not affine_matrix is None:
                pj = np.dot(affine_matrix, np.matrix([joint[0], joint[1], 1]).reshape(3, 1))
            else:
                pj = joint[0], joint[1]
            x, y = int(pj[1]), int(pj[0])# int(min(pj[1], 255)), int(min(pj[0], 191))
            xx, yy = circle(x, y, radius=radius, shape=img_size)
            colors[xx, yy] = kptcolors[i]
        return colors

    def get_joints(self, keypoints_path, affine_matrix=None, coeffs=None):
        with open(keypoints_path, 'r') as f:
            keypoints_data = json.load(f)
        if len(keypoints_data['people']) == 0:
            self.keypoints = np.zeros((18,3))
        else:
            self.keypoints = np.array(keypoints_data['people'][0]['pose_keypoints_2d']).reshape(-1,3)
        color_joint = self.draw_pose_from_cords(self.keypoints, (256, 192), affine_matrix, coeffs)
        return color_joint

    def valid_joints(self, joint):
        return (joint >= 0.1).all()

    def get_crop(self, bpart, order, wh, o_w, o_h, ar = 1.0):
        joints = self.keypoints
        bpart_indices = [order.index(b) for b in bpart]
        part_src = np.float32(joints[bpart_indices][:, :2])
        # fall backs
        if not self.valid_joints(joints[bpart_indices][:, 2]):
            if bpart[0] == "lhip" and bpart[1] == "lknee":
                bpart = ["lhip"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "rhip" and bpart[1] == "rknee":
                bpart = ["rhip"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            # elif bpart[0] == "lknee" and bpart[1] == "lankle":
            #     bpart = ["lknee"]
            #     bpart_indices = [order.index(b) for b in bpart]
            #     part_src = np.float32(joints[bpart_indices][:,:2])
            # elif bpart[0] == "rknee" and bpart[1] == "rankle":
            #     bpart = ["rknee"]
            #     bpart_indices = [order.index(b) for b in bpart]
            #     part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "lshoulder" and bpart[1] == "rshoulder" and bpart[2] == "cnose":
                bpart = ["lshoulder", "rshoulder", "rshoulder"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])

        if not self.valid_joints(joints[bpart_indices][:, 2]):
                return None, None
        part_src[:, 0] = part_src[:, 0] + 32                    # correct axis by adding pad size 

        if part_src.shape[0] == 1:
            # leg fallback
            a = part_src[0]
            b = np.float32([a[0],o_h - 1])
            part_src = np.float32([a,b])

        if part_src.shape[0] == 4:
            pass
        elif part_src.shape[0] == 3:
            # lshoulder, rshoulder, cnose
            if bpart == ["lshoulder", "rshoulder", "rshoulder"]:
                segment = part_src[1] - part_src[0]
                normal = np.array([-segment[1],segment[0]])
                if normal[1] > 0.0:
                    normal = -normal

                a = part_src[0] + normal
                b = part_src[0]
                c = part_src[1]
                d = part_src[1] + normal
                part_src = np.float32([a,b,c,d])
            else:
                assert bpart == ["lshoulder", "rshoulder", "cnose"]
                neck = 0.5*(part_src[0] + part_src[1])
                neck_to_nose = part_src[2] - neck
                part_src = np.float32([neck + 2*neck_to_nose, neck])

                # segment box
                segment = part_src[1] - part_src[0]
                normal = np.array([-segment[1],segment[0]])
                alpha = 1.0 / 2.0
                a = part_src[0] + alpha*normal
                b = part_src[0] - alpha*normal
                c = part_src[1] - alpha*normal
                d = part_src[1] + alpha*normal
                #part_src = np.float32([a,b,c,d])
                part_src = np.float32([b,c,d,a])
        else:
            assert part_src.shape[0] == 2

            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1],segment[0]])
            alpha = ar / 2.0
            a = part_src[0] + alpha*normal
            b = part_src[0] - alpha*normal
            c = part_src[1] - alpha*normal
            d = part_src[1] + alpha*normal
            part_src = np.float32([a,b,c,d])

        dst = np.float32([[0.0,0.0],[0.0,1.0],[1.0,1.0],[1.0,0.0]])
        part_dst = np.float32(wh * dst)

        M = cv2.getPerspectiveTransform(part_src, part_dst)
        M_inv = cv2.getPerspectiveTransform(part_dst,part_src)
        return M, M_inv

    def normalize(self, upper_img, lower_img, upper_clothes_mask, lower_clothes_mask, box_factor):
        h, w = upper_img.shape[:2]
        o_h, o_w = h, w
        h = h // 2**box_factor
        w = w // 2**box_factor
        wh = np.array([w, h])
        wh = np.expand_dims(wh, 0)

        bparts = [
                ["lshoulder","lhip","rhip","rshoulder"],
                ["lshoulder", "rshoulder", "cnose"],
                ["lshoulder","lelbow"],
                ["lelbow", "lwrist"],
                ["rshoulder","relbow"],
                ["relbow", "rwrist"],
                ["lhip", "lknee"],
                ["lknee", "lankle"],
                ["rhip", "rknee"],
                ["rknee", "rankle"]]

        order = ['cnose', 'cneck', 'rshoulder', 'relbow', 'rwrist', 'lshoulder', 
                'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 
                'lankle', 'reye', 'leye', 'rear', 'lear']
        ar = 0.5

        part_imgs = list()
        part_imgs_lower = list()
        part_clothes_masks = list()
        part_clothes_masks_lower = list()
        M_invs = list()
        denorm_hand_masks = list()

        denorm_upper_img = np.zeros_like(upper_img)
        denorm_lower_img = np.zeros_like(upper_img)

        for ii, bpart in enumerate(bparts):
            part_img = np.zeros((h, w, 3)).astype(np.uint8)
            part_img_lower = np.zeros((h,w,3)).astype(np.uint8)
            part_clothes_mask = np.zeros((h,w,3)).astype(np.uint8)
            part_clothes_mask_lower = np.zeros((h,w,3)).astype(np.uint8)
            M, M_inv = self.get_crop(bpart, order, wh, o_w, o_h, ar)

            if M is not None:
                part_img = cv2.warpPerspective(upper_img, M, (w,h), borderMode = cv2.BORDER_REPLICATE)
                part_clothes_mask = cv2.warpPerspective(upper_clothes_mask, M, (w,h), borderMode = cv2.BORDER_REPLICATE)
                
                denorm_patch = cv2.warpPerspective(part_img, M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)
                denorm_clothes_mask_patch = cv2.warpPerspective(part_clothes_mask, M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)[...,0:1]
                denorm_clothes_mask_patch = (denorm_clothes_mask_patch==255).astype(np.uint8)

                denorm_upper_img = denorm_patch * denorm_clothes_mask_patch + denorm_upper_img * (1-denorm_clothes_mask_patch)

                if ii >= 6:
                    part_img_lower = cv2.warpPerspective(lower_img, M, (w,h), borderMode = cv2.BORDER_REPLICATE)
                    part_clothes_mask_lower = cv2.warpPerspective(lower_clothes_mask, M, (w,h), borderMode = cv2.BORDER_REPLICATE)

                    denorm_patch_lower = cv2.warpPerspective(part_img_lower, M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)
                    denorm_clothes_mask_patch_lower = cv2.warpPerspective(part_clothes_mask_lower, M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)[...,0:1]
                    denorm_clothes_mask_patch_lower = (denorm_clothes_mask_patch_lower==255).astype(np.uint8)

                    denorm_upper_img = denorm_patch_lower * denorm_clothes_mask_patch_lower + denorm_upper_img * (1-denorm_clothes_mask_patch_lower)

                M_invs.append(M_inv[np.newaxis,...])
            else:
                M_invs.append(np.zeros((1,3,3),dtype=np.float32))

            if ii >= 2 and ii <= 5:
                if M is not None:
                    denorm_hand_masks.append(denorm_clothes_mask_patch)
                else:
                    denorm_hand_masks.append(np.zeros_like(upper_img)[...,0:1])
            # if ii == 7 or ii ==9:
            #     if M is not None:
            #         denorm_leg_masks.append(denorm_clothes_mask_patch)
            #     else:
            #         denorm_leg_masks.append(np.zeros_like(lower_img)[...,0:1])

            part_imgs.append(part_img)
            part_clothes_masks.append(part_clothes_mask)
            if ii >= 6:
                part_imgs_lower.append(part_img_lower)
                part_clothes_masks_lower.append(part_clothes_mask_lower)

        img = np.concatenate(part_imgs, axis = 2)
        img_lower = np.concatenate(part_imgs_lower, axis=2)
        clothes_masks = np.concatenate(part_clothes_masks, axis=2)
        clothes_masks_lower = np.concatenate(part_clothes_masks_lower, axis=2)
        M_invs = np.concatenate(M_invs, axis=0)

        return img, img_lower, denorm_upper_img, denorm_lower_img, M_invs, denorm_hand_masks, clothes_masks, clothes_masks_lower

    def __getitem__(self, idx):
        image, pose, norm_img, norm_img_lower, denorm_upper_img, denorm_lower_img, M_invs, gt_parsing, \
            denorm_hand_masks, norm_clothes_masks, norm_clothes_masks_lower, retain_mask = self._load_raw_image(self._raw_idx[idx])

        image = image.transpose(2, 0, 1)                    # HWC => CHW
        pose = pose.transpose(2, 0, 1)                      # HWC => CHW
        norm_img = norm_img.transpose(2, 0, 1)
        norm_img_lower = norm_img_lower.transpose(2,0,1)
        # norm_pose = norm_pose.transpose(2, 0, 1)
        denorm_upper_img = denorm_upper_img.transpose(2,0,1)
        denorm_lower_img = denorm_lower_img.transpose(2,0,1)

        norm_clothes_masks = norm_clothes_masks.transpose(2,0,1)
        norm_clothes_masks_lower = norm_clothes_masks_lower.transpose(2,0,1)

        # upper_clothes_mask = upper_clothes_mask.transpose(2,0,1)
        # lower_clothes_mask = lower_clothes_mask.transpose(2,0,1)
        gt_parsing = gt_parsing.transpose(2,0,1)

        retain_mask = retain_mask.transpose(2,0,1)

        # concat the pose and img since they often binded together
        # norm_img = np.concatenate((norm_img, norm_pose), axis=0)

    
        denorm_random_mask = np.zeros((256,256,1),dtype=np.uint8)
        # random.seed(5)
        random.seed(1)

        if random.random() < 0.4:
            for mask in denorm_hand_masks:
                if random.random() < 0.5:
                    denorm_random_mask += mask
        # if random.random() < 0.4:
        #     for mask in denorm_leg_masks:
        #         if random.random() < 0.5:
        #             denorm_random_mask += mask

        if random.random() < 0.9:
            fname = self._random_mask_acgpn_fnames[self._raw_idx[idx]%self._mask_acgpn_numbers]
            random_mask = cv2.imread(fname)[...,0:1]
            random_mask = cv2.resize(random_mask,(256,256))[...,np.newaxis]
            denorm_random_mask += random_mask

        denorm_random_mask = (denorm_random_mask>0).astype(np.uint8)
        denorm_random_mask = denorm_random_mask.transpose(2,0,1)

        denorm_upper_img_erase = denorm_upper_img * (1-denorm_random_mask)
        denorm_upper_mask = (np.sum(denorm_upper_img_erase, axis=0, keepdims=True)>0).astype(np.uint8)
        denorm_lower_img_erase = denorm_lower_img * (1-denorm_random_mask)
        denorm_lower_mask = (np.sum(denorm_lower_img_erase, axis=0, keepdims=True)>0).astype(np.uint8)

        # assert isinstance(image, np.ndarray)
        # assert list(image.shape) == self.image_shape
        # assert image.dtype == np.uint8

        # return image.copy(), pose.copy(), sem.copy(), norm_img.copy(), denorm_upper_img.copy(), denorm_lower_img.copy(), \
        #         M_invs.copy(), upper_clothes_mask.copy(), lower_clothes_mask.copy(), denorm_random_mask.copy(), \
        #         denorm_upper_mask.copy(), denorm_lower_mask.copy(), norm_clothes_masks.copy()
        # return image.copy(), pose.copy(), sem.copy(), norm_img.copy(), denorm_upper_img_erase.copy(), denorm_lower_img_erase.copy(), \
        #         M_invs.copy(), upper_clothes_mask.copy(), lower_clothes_mask.copy(), \
        #         denorm_upper_mask.copy(), denorm_lower_mask.copy(), norm_clothes_masks.copy(), retain_mask.copy()
        return image.copy(), pose.copy(), norm_img.copy(), norm_img_lower.copy(), denorm_upper_img_erase.copy(), denorm_lower_img_erase.copy(), \
                M_invs.copy(), gt_parsing.copy(), denorm_upper_mask.copy(), denorm_lower_mask.copy(), \
                norm_clothes_masks.copy(), norm_clothes_masks_lower.copy(), retain_mask.copy()


############# Dataset for full body model testing ############
class UvitonDatasetV19_test(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            # dataset_list = ['UPT_subset1_256_192', 'UPT_subset2_256_192', 'Deepfashion_256_192', 'MPV_256_192']
            dataset_list = ['UPT_subset1_256_192', 'UPT_subset2_256_192']
            self._image_fnames = []
            self._kpt_fnames = []
            self._parsing_fnames = []

            self._clothes_image_fnames = []
            self._clothes_kpt_fnames = []
            self._clothes_parsing_fnames = []

            for dataset in dataset_list:
                # txt_path = os.path.join(self._path, dataset, 'test_pairs_front_half_list_shuffle_0508.txt')
                txt_path = os.path.join(self._path, dataset, 'test_pairs_front_list_shuffle_0508.txt')
                with open(txt_path, 'r') as f:
                    for line in f.readlines():
                        person, clothes = line.strip().split()
                        self._image_fnames.append(os.path.join(dataset,'image',person))
                        self._kpt_fnames.append(os.path.join(dataset,'keypoints',person.replace('.jpg', '_keypoints.json')))

                        self._clothes_image_fnames.append(os.path.join(dataset,'image',clothes))
                        self._clothes_kpt_fnames.append(os.path.join(dataset,'keypoints',clothes.replace('.jpg', '_keypoints.json')))

                        if dataset == 'MPV_256_192':
                            self._parsing_fnames.append(os.path.join(dataset,'parsing',person.replace('.jpg','.png')))
                            self._clothes_parsing_fnames.append(os.path.join(dataset,'parsing',clothes.replace('.jpg','.png')))
                        else:
                            self._parsing_fnames.append(os.path.join(dataset,'parsing',person.replace('.jpg','_label.png')))
                            self._clothes_parsing_fnames.append(os.path.join(dataset,'parsing',clothes.replace('.jpg','_label.png')))
        else:
            raise IOError('Path must point to a directory or zip')

        self._vis_index = list(range(64))    # vis_index 

        PIL.Image.init()
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        im_shape = list((self._load_raw_image(0))[0].shape)
        raw_shape = [len(self._image_fnames)] + [im_shape[2], im_shape[0], im_shape[1]]
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)


    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        # load images --> range [0, 255]
        fname = self._image_fnames[raw_idx]
        person_name = fname
        f = os.path.join(self._path, fname)
        self.image = np.array(PIL.Image.open(f))

        im_shape = self.image.shape
        # padding to same size
        h, w = im_shape[0], im_shape[1]
        left_padding = (h-w) // 2
        right_padding = h-w-left_padding

        image = np.pad(self.image,((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(255,255))

        # load keypoints --> range [0, 1]
        fname = self._kpt_fnames[raw_idx]
        kpt = os.path.join(self._path, fname)
        pose, keypoints = self.get_joints(kpt) # self.cords_to_map(kpt, im_shape[:2])
        pose = np.pad(pose,((0,0),(left_padding,right_padding),(0,0)),'constant',constant_values=(0,0))
        keypoints[:,0] += left_padding

        # load upper_cloth and lower body
        fname = self._parsing_fnames[raw_idx]
        f = os.path.join(self._path, fname)
        parsing = cv2.imread(f)[..., 0:1]
        parsing = np.pad(parsing,((0,0),(left_padding,right_padding),(0,0)),'constant',constant_values=(0,0))

        palm_mask = self.get_palm(keypoints, parsing)
        head_mask = (parsing==1).astype(np.uint8) + (parsing==4).astype(np.uint8) + \
                    (parsing==2).astype(np.uint8) + (parsing==13).astype(np.uint8)
        shoes_mask = (parsing==18).astype(np.uint8) + (parsing==19).astype(np.uint8)
        
        lower_clothes_mask = (parsing==9).astype(np.uint8) + (parsing==12).astype(np.uint8) + \
                            (parsing==6).astype(np.uint8)
        lower_clothes_image = lower_clothes_mask * image        

        image = image * (palm_mask+head_mask+shoes_mask)

        fname = self._clothes_image_fnames[raw_idx]
        clothes_name = fname
        f = os.path.join(self._path, fname)
        self.clothes = np.array(PIL.Image.open(f))
        clothes = np.pad(self.clothes,((0,0),(left_padding,right_padding),(0,0)), 'constant',constant_values=(255,255))

        fname = self._clothes_kpt_fnames[raw_idx]
        kpt = os.path.join(self._path, fname)
        clothes_pose, clothes_keypoints = self.get_joints(kpt) # self.cords_to_map(kpt, im_shape[:2])
        clothes_pose = np.pad(clothes_pose,((0,0),(left_padding,right_padding),(0,0)),'constant',constant_values=(0,0))
        clothes_keypoints[:,0] += left_padding

        fname = self._clothes_parsing_fnames[raw_idx]
        f = os.path.join(self._path, fname)
        clothes_parsing = cv2.imread(f)[..., 0:1]
        clothes_parsing = np.pad(clothes_parsing,((0,0),(left_padding,right_padding),(0,0)),'constant',constant_values=(0,0))

        upper_clothes_mask = (clothes_parsing==5).astype(np.uint8) + (clothes_parsing==6).astype(np.uint8) + \
                            (clothes_parsing==7).astype(np.uint8)
        upper_clothes_image = upper_clothes_mask * clothes

        upper_clothes_mask_rgb = np.concatenate([upper_clothes_mask,upper_clothes_mask,upper_clothes_mask],axis=2)
        lower_clothes_mask_rgb = np.concatenate([lower_clothes_mask,lower_clothes_mask,lower_clothes_mask],axis=2)
        upper_clothes_mask_rgb = upper_clothes_mask_rgb * 255
        lower_clothes_mask_rgb = lower_clothes_mask_rgb * 255

        upper_pose = clothes_pose
        lower_pose = pose
        upper_keypoints = clothes_keypoints
        lower_keypoints = keypoints

        norm_img, norm_pose, denorm_upper_img, denorm_lower_img = self.normalize(upper_clothes_image, lower_clothes_image, \
                upper_clothes_mask_rgb, lower_clothes_mask_rgb, upper_pose, lower_pose, upper_keypoints, lower_keypoints, 2)

        return image, pose, norm_img, norm_pose, denorm_upper_img, denorm_lower_img, person_name, clothes_name


    def _load_raw_labels(self):
        fname = 'dataset.json'
        if not os.path.exists(os.path.join(self._path, fname)):
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def cords_to_map(self, keypoints_path, img_size, old_size=None,
                     affine_matrix=None, coeffs=None, sigma=6):
        with open(keypoints_path, 'r') as f:
            keypoints_data = json.load(f)
        if len(keypoints_data['people']) == 0:
            self.cords = np.zeros((18, 3))
        else:
            self.cords = np.array(keypoints_data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
        old_size = img_size if old_size is None else old_size
        self.cords = self.cords.astype(float)
        result = np.zeros(img_size + (self.cords.shape[0],), dtype='float32')
        for i, (x, y, score) in enumerate(self.cords):
            if score < 0.1:
                continue
            x = x / old_size[0] * img_size[0]
            y = y / old_size[1] * img_size[1]
            if affine_matrix is not None:
                point_ = np.dot(affine_matrix, np.matrix([x, y, 1]).reshape(3, 1))
                x = int(point_[0])
                y = int(point_[1])
            else:
                x = int(x)
                y = int(y)
            if coeffs is not None:
                a, b, c, d, e, f, g, h = coeffs
                x = int((a * x + b * y + c) / (g * x + h * y + 1))
                y = int((d * x + e * y + f) / (g * x + h * y + 1))
            xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
            result[..., i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

        return result
    
    ############################ get palm mask #########################################

    def get_mask_from_kps(self, kps, img_h, img_w):
        rles = maskUtils.frPyObjects(kps, img_h, img_w)
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)[...,np.newaxis].astype(np.float32)
        mask = mask * 255.0
        return mask

    def get_rectangle_mask(self, a, b, c, d, img_h, img_w):
        x1, y1 = a + (b-d)/4,   b + (c-a)/4
        x2, y2 = a - (b-d)/4,   b - (c-a)/4

        x3, y3 = c + (b-d)/4,   d + (c-a)/4
        x4, y4 = c - (b-d)/4,   d - (c-a)/4

        kps  = [x1,y1,x2,y2]

        v0_x, v0_y = c-a,   d-b
        v1_x, v1_y = x3-x1, y3-y1
        v2_x, v2_y = x4-x1, y4-y1

        cos1 = (v0_x*v1_x+v0_y*v1_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v1_x*v1_x+v1_y*v1_y))
        cos2 = (v0_x*v2_x+v0_y*v2_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v2_x*v2_x+v2_y*v2_y))

        if cos1<cos2:
            kps.extend([x3,y3,x4,y4])
        else:
            kps.extend([x4,y4,x3,y3])

        kps = np.array(kps).reshape(1,-1).tolist()
        mask = self.get_mask_from_kps(kps, img_h=img_h, img_w=img_w)

        return mask
    
    def get_hand_mask(self, hand_keypoints):
        s_x,s_y,s_c = hand_keypoints[0]
        e_x,e_y,e_c = hand_keypoints[1]
        w_x,w_y,w_c = hand_keypoints[2]

        h, w = 256, 256
        up_mask = np.ones((256,256,1),dtype=np.float32)
        bottom_mask = np.ones((256,256,1),dtype=np.float32)
        if s_c > 0.1 and e_c > 0.1:
            up_mask = self.get_rectangle_mask(s_x, s_y, e_x, e_y, h, w)
            kernel = np.ones((25,25),np.uint8)
            up_mask = cv2.dilate(up_mask,kernel,iterations=1)
            up_mask = (up_mask > 0).astype(np.float32)[...,np.newaxis]
        if e_c > 0.1 and w_c > 0.1:
            bottom_mask = self.get_rectangle_mask(e_x, e_y, w_x, w_y, h, w)
            kernel = np.ones((15,15),np.uint8)
            bottom_mask = cv2.dilate(bottom_mask,kernel,iterations=1)
            bottom_mask = (bottom_mask > 0).astype(np.float32)[...,np.newaxis]

        return up_mask, bottom_mask

    def get_palm_mask(self, hand_mask, hand_up_mask, hand_bottom_mask):
        inter_up_mask = ((hand_mask + hand_up_mask) == 2).astype(np.float32)
        hand_mask = hand_mask - inter_up_mask
        inter_bottom_mask = ((hand_mask+hand_bottom_mask)==2).astype(np.float32)
        palm_mask = hand_mask - inter_bottom_mask

        return palm_mask

    def get_palm(self, keypoints, parsing):
        left_hand_keypoints = keypoints[[5,6,7],:].copy()
        right_hand_keypoints = keypoints[[2,3,4],:].copy()

        left_hand_up_mask, left_hand_botton_mask = self.get_hand_mask(left_hand_keypoints)
        right_hand_up_mask, right_hand_botton_mask = self.get_hand_mask(right_hand_keypoints)

        # mask refined by parsing
        left_hand_mask = (parsing == 14).astype(np.float32)
        right_hand_mask = (parsing == 15).astype(np.float32)
        left_palm_mask = self.get_palm_mask(left_hand_mask, left_hand_up_mask, left_hand_botton_mask)
        right_palm_mask = self.get_palm_mask(right_hand_mask, right_hand_up_mask, right_hand_botton_mask)
        palm_mask = ((left_palm_mask + right_palm_mask) > 0).astype(np.uint8)

        return palm_mask

    ######################################################################################

    def draw_pose_from_cords(self, pose_joints, img_size, affine_matrix=None,
                             coeffs=None, radius=2, draw_joints=True):
        colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
        # mask = np.zeros(shape=img_size, dtype=np.uint8)
        if draw_joints:
            for i, p in enumerate(limbseq):
                f, t = p[0]-1, p[1]-1
                from_missing = pose_joints[f][2] < 0.1
                to_missing = pose_joints[t][2] < 0.1
                if from_missing or to_missing:
                    continue
                if not affine_matrix is None:
                    pf = np.dot(affine_matrix, np.matrix([pose_joints[f][0], pose_joints[f][1], 1]).reshape(3, 1))
                    pt = np.dot(affine_matrix, np.matrix([pose_joints[t][0], pose_joints[t][1], 1]).reshape(3, 1))
                else:
                    pf = pose_joints[f][0], pose_joints[f][1]
                    pt = pose_joints[t][0], pose_joints[t][1]
                fx, fy = pf[1], pf[0]# max(pf[1], 0), max(pf[0], 0)
                tx, ty = pt[1], pt[0]# max(pt[1], 0), max(pt[0], 0)
                fx, fy = int(fx), int(fy)# int(min(fx, 255)), int(min(fy, 191))
                tx, ty = int(tx), int(ty)# int(min(tx, 255)), int(min(ty, 191))
                # xx, yy, val = line_aa(fx, fy, tx, ty)
                # colors[xx, yy] = np.expand_dims(val, 1) * kptcolors[i] # 255
                cv2.line(colors, (fy, fx), (ty, tx), kptcolors[i], 2)
                # mask[xx, yy] = 255

        for i, joint in enumerate(pose_joints):
            if pose_joints[i][2] < 0.1:
                continue
            if not affine_matrix is None:
                pj = np.dot(affine_matrix, np.matrix([joint[0], joint[1], 1]).reshape(3, 1))
            else:
                pj = joint[0], joint[1]
            x, y = int(pj[1]), int(pj[0])# int(min(pj[1], 255)), int(min(pj[0], 191))
            xx, yy = circle(x, y, radius=radius, shape=img_size)
            colors[xx, yy] = kptcolors[i]
            # mask[xx, yy] = 255

        # colors = colors * 1./255
        # mask = mask * 1./255
        
        return colors

    def get_joints(self, keypoints_path, affine_matrix=None, coeffs=None):
        with open(keypoints_path, 'r') as f:
            keypoints_data = json.load(f)
        if len(keypoints_data['people']) == 0:
            keypoints = np.zeros((18,3))
        else:
            keypoints = np.array(keypoints_data['people'][0]['pose_keypoints_2d']).reshape(-1,3)
        # joints = self.kp_to_map(img_sz=(192,256), kps=keypoints)
        color_joint = self.draw_pose_from_cords(keypoints, (256, 192), affine_matrix, coeffs)
        return color_joint, keypoints

    def valid_joints(self, joint):
        return (joint >= 0.1).all()

    def get_crop(self, keypoints, bpart, order, wh, o_w, o_h, ar = 1.0):
        joints = keypoints
        bpart_indices = [order.index(b) for b in bpart]
        part_src = np.float32(joints[bpart_indices][:, :2])
        # fall backs
        if not self.valid_joints(joints[bpart_indices][:, 2]):
            if bpart[0] == "lhip" and bpart[1] == "lknee":      # 鏈塰ip鍏抽敭鐐逛絾鏄病鏈塳nee鍏抽敭鐐?
                bpart = ["lhip"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "rhip" and bpart[1] == "rknee":    #銆€宸﹁竟鍚岢�悄1�7
                bpart = ["rhip"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "lknee" and bpart[1] == "lankle":
                bpart = ["lknee"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "rknee" and bpart[1] == "rankle":
                bpart = ["rknee"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "lshoulder" and bpart[1] == "rshoulder" and bpart[2] == "cnose": # 娌℃湁宸﹁偐鍙宠偄1�7,榧诲瓙杩欑粍鍖哄煄1�7
                bpart = ["lshoulder", "rshoulder", "rshoulder"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])

        if not self.valid_joints(joints[bpart_indices][:, 2]):
                return None, None
        # part_src[:, 0] = part_src[:, 0] + 32                    # correct axis by adding pad size 

        if part_src.shape[0] == 1:
            # leg fallback
            a = part_src[0]
            b = np.float32([a[0],o_h - 1])
            part_src = np.float32([a,b])

        if part_src.shape[0] == 4:
            pass
        elif part_src.shape[0] == 3:
            # lshoulder, rshoulder, cnose
            if bpart == ["lshoulder", "rshoulder", "rshoulder"]:
                segment = part_src[1] - part_src[0]
                normal = np.array([-segment[1],segment[0]])
                if normal[1] > 0.0:
                    normal = -normal

                a = part_src[0] + normal
                b = part_src[0]
                c = part_src[1]
                d = part_src[1] + normal
                part_src = np.float32([a,b,c,d])
            else:
                assert bpart == ["lshoulder", "rshoulder", "cnose"]
                neck = 0.5*(part_src[0] + part_src[1])
                neck_to_nose = part_src[2] - neck
                part_src = np.float32([neck + 2*neck_to_nose, neck])

                # segment box
                segment = part_src[1] - part_src[0]
                normal = np.array([-segment[1],segment[0]])
                alpha = 1.0 / 2.0
                a = part_src[0] + alpha*normal
                b = part_src[0] - alpha*normal
                c = part_src[1] - alpha*normal
                d = part_src[1] + alpha*normal
                #part_src = np.float32([a,b,c,d])
                part_src = np.float32([b,c,d,a])
        else:
            assert part_src.shape[0] == 2

            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1],segment[0]])
            alpha = ar / 2.0
            a = part_src[0] + alpha*normal
            b = part_src[0] - alpha*normal
            c = part_src[1] - alpha*normal
            d = part_src[1] + alpha*normal
            part_src = np.float32([a,b,c,d])

        dst = np.float32([[0.0,0.0],[0.0,1.0],[1.0,1.0],[1.0,0.0]])
        part_dst = np.float32(wh * dst)

        M = cv2.getPerspectiveTransform(part_src, part_dst)
        M_inv = cv2.getPerspectiveTransform(part_dst,part_src)
        return M, M_inv

    def normalize(self, upper_img, lower_img, upper_clothes_mask, lower_clothes_mask, upper_pose, lower_pose, \
                    upper_keypoints, lower_keypoints, box_factor):

        h, w = upper_img.shape[:2]
        o_h, o_w = h, w
        h = h // 2**box_factor
        w = w // 2**box_factor
        wh = np.array([w, h])
        wh = np.expand_dims(wh, 0)

        bparts = [
                ["lshoulder","lhip","rhip","rshoulder"],
                ["lshoulder", "rshoulder", "cnose"],
                ["lshoulder","lelbow"],
                ["lelbow", "lwrist"],
                ["rshoulder","relbow"],
                ["relbow", "rwrist"],
                ["lhip", "lknee"],
                ["lknee", "lankle"],
                ["rhip", "rknee"],
                ["rknee", "rankle"]]

        order = ['cnose', 'cneck', 'rshoulder', 'relbow', 'rwrist', 'lshoulder', 
                'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 
                'lankle', 'reye', 'leye', 'rear', 'lear']
        ar = 0.5

        part_imgs = list()
        part_stickmen = list()

        denorm_upper_img = np.zeros_like(upper_img)
        denorm_lower_img = np.zeros_like(upper_img)
        kernel = np.ones((5,5),np.uint8)

        for ii, bpart in enumerate(bparts):
            part_img = np.zeros((h, w, 3)).astype(np.uint8)
            part_stickman = np.zeros((h, w, 3)).astype(np.uint8)
            part_clothes_mask = np.zeros((h, w, 3)).astype(np.uint8)
            
            upper_M, _ = self.get_crop(upper_keypoints, bpart, order, wh, o_w, o_h, ar)
            lower_M, lower_M_inv = self.get_crop(lower_keypoints, bpart, order, wh, o_w, o_h, ar)

            if ii < 6:
                if upper_M is not None:
                    part_img = cv2.warpPerspective(upper_img, upper_M, (w,h), borderMode = cv2.BORDER_REPLICATE)
                    part_stickman = cv2.warpPerspective(upper_pose, upper_M, (w,h), borderMode = cv2.BORDER_REPLICATE)
                    part_clothes_mask = cv2.warpPerspective(upper_clothes_mask, upper_M, (w,h), borderMode = cv2.BORDER_REPLICATE)
            else:
                if lower_M is not None:
                    part_img = cv2.warpPerspective(lower_img, lower_M, (w,h), borderMode = cv2.BORDER_REPLICATE)
                    part_stickman = cv2.warpPerspective(lower_pose, lower_M, (w,h), borderMode = cv2.BORDER_REPLICATE)
                    part_clothes_mask = cv2.warpPerspective(lower_clothes_mask, lower_M, (w,h), borderMode = cv2.BORDER_REPLICATE)

            if lower_M_inv is not None:
                denorm_patch = cv2.warpPerspective(part_img, lower_M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)
                denorm_clothes_mask_patch = cv2.warpPerspective(part_clothes_mask, lower_M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)
                if ii < 6:
                    denorm_clothes_mask_patch = cv2.erode(denorm_clothes_mask_patch, kernel, iterations=1)
                denorm_clothes_mask_patch = denorm_clothes_mask_patch[...,0:1]
                denorm_clothes_mask_patch = (denorm_clothes_mask_patch==255).astype(np.uint8)
                
                if ii < 6:
                    denorm_upper_img = denorm_patch * denorm_clothes_mask_patch + denorm_upper_img * (1-denorm_clothes_mask_patch)
                else:
                    denorm_lower_img = denorm_patch * denorm_clothes_mask_patch + denorm_lower_img * (1-denorm_clothes_mask_patch)
                
            part_imgs.append(part_img)
            part_stickmen.append(part_stickman)

        img = np.concatenate(part_imgs, axis = 2)
        stickman = np.concatenate(part_stickmen, axis = 2)

        return img, stickman, denorm_upper_img, denorm_lower_img

    def __getitem__(self, idx):
        image, pose, norm_img, norm_pose, denorm_upper_img, denorm_lower_img, person_name, clothes_name = self._load_raw_image(self._raw_idx[idx])

        image = image.transpose(2, 0, 1)                    # HWC => CHW
        pose = pose.transpose(2, 0, 1)                      # HWC => CHW
        norm_img = norm_img.transpose(2, 0, 1)
        norm_pose = norm_pose.transpose(2, 0, 1)
        denorm_upper_img = denorm_upper_img.transpose(2,0,1)
        denorm_lower_img = denorm_lower_img.transpose(2,0,1)

        # concat the pose and img since they often binded together
        norm_img = np.concatenate((norm_img, norm_pose), axis=0)
    
        denorm_upper_mask = (np.sum(denorm_upper_img, axis=0, keepdims=True)>0).astype(np.uint8)
        denorm_lower_mask = (np.sum(denorm_lower_img, axis=0, keepdims=True)>0).astype(np.uint8)

        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8

        return image.copy(), pose.copy(), norm_img.copy(), denorm_upper_img.copy(), denorm_lower_img.copy(), \
                denorm_upper_mask.copy(), denorm_lower_mask.copy(), person_name, clothes_name

