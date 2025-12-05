import torch.utils.data as data
from PIL import Image
import numpy as np
from imageio import imread
import random
import torch
import time
import cv2
from PIL import ImageFile
from transform_list import RandomCropNumpy, EnhancedCompose, RandomColor, RandomHorizontalFlip, ArrayToTensorNumpy, Normalize, CropNumpy
from torchvision import transforms
import pdb
import os 

ImageFile.LOAD_TRUNCATED_IMAGES = True

def _is_pil_image(img):
    return isinstance(img, Image.Image)

class MyDataset(data.Dataset):
    # ... (کلاس MyDataset بدون تغییر در منطق) ...
    # (همه FIX‌ها در این کلاس حفظ می‌شوند)
    def __init__(self, args, train=True, return_filename=False):
        # ... (init logic remains the same) ...
        self.use_dense_depth = args.use_dense_depth
        if train is True:
            if args.dataset == 'KITTI':
                self.datafile = args.trainfile_kitti
                self.angle_range = (-1, 1)
                self.depth_scale = 256.0
            elif args.dataset == 'NYU':
                self.datafile = args.trainfile_nyu
                self.angle_range = (-2.5, 2.5)
                self.depth_scale = 1000.0
                args.height = 416
                args.width = 544
        else:
            if args.dataset == 'KITTI':
                self.datafile = args.testfile_kitti
                self.depth_scale = 256.0
            elif args.dataset == 'NYU':
                self.datafile = args.testfile_nyu
                self.depth_scale = 1000.0
                args.height = 416
                args.width = 544
        
        self.train = train
        self.transform = Transformer(args)
        self.args = args

        self.data_path = self.args.data_path 
        
        self.return_filename = return_filename
        with open(self.datafile, 'r') as f:
            self.fileset = f.readlines()
        self.fileset = sorted(self.fileset)
    
    def __getitem__(self, index):
        line = self.fileset[index].strip()
        divided_file = [f.strip() for f in line.split() if f.strip()]

        if len(divided_file) < 2:
            raise IndexError(f"List line is incomplete: {line}")

        rgb_name = divided_file[0]
        gt_name = divided_file[1]

        rgb_file = os.path.join(self.data_path, rgb_name)
        rgb = Image.open(rgb_file)
        
        gt = False
        gt_dense = False

        filename = rgb_name.split(".")[0].strip()

        # ... (Train/Test logic and Cropping remain the same) ...
        if self.train is False:
            if self.args.dataset == 'KITTI':
                if gt_name != 'None':
                    gt_file = os.path.join(self.data_path, 'data_depth_annotated', gt_name)
                    gt = Image.open(gt_file)
                    if self.use_dense_depth is True:
                        if len(divided_file) > 2:
                            gt_dense_file = os.path.join(self.data_path, 'data_depth_annotated', divided_file[2])
                        else:
                            gt_dense_file = gt_file
                        gt_dense = Image.open(gt_dense_file)
            elif self.args.dataset == 'NYU':
                gt_file = os.path.join(self.data_path, gt_name)
                gt = Image.open(gt_file)
                if self.use_dense_depth is True:
                    if len(divided_file) > 2:
                        gt_dense_file = os.path.join(self.data_path, divided_file[2])
                    else:
                        gt_dense_file = gt_file
                    gt_dense = Image.open(gt_dense_file)
        else:
            angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
            if self.args.dataset == 'KITTI':
                gt_file = os.path.join(self.data_path, 'data_depth_annotated', gt_name)
                if self.use_dense_depth is True:
                    if len(divided_file) > 2:
                        gt_dense_file = os.path.join(self.data_path, 'data_depth_annotated', divided_file[2])
                    else:
                        gt_dense_file = gt_file
            elif self.args.dataset == 'NYU':
                gt_file = os.path.join(self.data_path, gt_name)
                if self.use_dense_depth is True:
                    if len(divided_file) > 2:
                        gt_dense_file = os.path.join(self.data_path, divided_file[2])
                    else:
                        gt_dense_file = gt_file
            
            gt = Image.open(gt_file)
            rgb = rgb.rotate(angle, resample=Image.BILINEAR)
            gt = gt.rotate(angle, resample=Image.NEAREST)
            if self.use_dense_depth is True:
                gt_dense = Image.open(gt_dense_file)
                gt_dense = gt_dense.rotate(angle, resample=Image.NEAREST)

        if self.args.dataset == 'KITTI':
            h = rgb.height
            w = rgb.width
            bound_left = (w - 1216) // 2
            bound_right = bound_left + 1216
            bound_top = h - 352
            bound_bottom = bound_top + 352
        elif self.args.dataset == 'NYU':
            if self.train is True:
                bound_left = 43
                bound_right = 608
                bound_top = 45
                bound_bottom = 472
            else:
                bound_left = 0
                bound_right = 640
                bound_top = 0
                bound_bottom = 480
        
        if (self.args.dataset == 'NYU' and (self.train is False) and (self.return_filename is False)):
            rgb = rgb.crop((40+20, 42+14, 616-12, 474-2))
        else:
            rgb = rgb.crop((bound_left, bound_top, bound_right, bound_bottom))

        rgb = np.asarray(rgb, dtype=np.float32) / 255.0

        if _is_pil_image(gt):
            gt = gt.crop((bound_left, bound_top, bound_right, bound_bottom))
            gt = (np.asarray(gt, dtype=np.float32)) / self.depth_scale
            gt = np.expand_dims(gt, axis=2)
            gt = np.clip(gt, 0, self.args.max_depth)
        
        if self.use_dense_depth is True:
            if _is_pil_image(gt_dense):
                gt_dense = gt_dense.crop((bound_left, bound_top, bound_right, bound_bottom))
                gt_dense = (np.asarray(gt_dense, dtype=np.float32)) / self.depth_scale
                gt_dense = np.expand_dims(gt_dense, axis=2)
                gt_dense = np.clip(gt_dense, 0, self.args.max_depth)
                gt_dense = gt_dense * (gt.max() / gt_dense.max())

        # اعمال Transforms
        rgb, gt, gt_dense = self.transform([rgb] + [gt] + [gt_dense], self.train)
        
        # --- DEBUG DATA CHECK ---
        if index == 0:
            print(f"\n========== DEBUG DATA CHECK (Index {index}) ==========")
            print(f"RGB Shape: {rgb.shape}")
            print(f"RGB Min: {rgb.min():.4f}, RGB Max: {rgb.max():.4f}")
            print(f"RGB Mean: {rgb.mean():.4f} (Goal: ~0.0 for normalized, ~0.5 for raw)")
            print(f"RGB Std:  {rgb.std():.4f} (Goal: ~1.0 for normalized, ~0.2 for raw)")
            print("====================================================\n")
        # ------------------------

        if self.return_filename is True:
            return rgb, gt, gt_dense, filename
        else:
            return rgb, gt, gt_dense

    def __len__(self):
        return len(self.fileset)


class Transformer(object):
    def __init__(self, args):
        if args.dataset == 'KITTI':
            self.train_transform = EnhancedCompose([
                RandomCropNumpy((args.height, args.width)),
                RandomHorizontalFlip(),
                [RandomColor(multiplier_range=(0.9, 1.1)), None, None],
                ArrayToTensorNumpy(),
                # FIX: حذف transforms.Normalize و استفاده از ساختار مستقل (اگرچه هنوز در لیست است)
                [Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
            self.test_transform = EnhancedCompose([
                CropNumpy((args.height, args.width)),
                ArrayToTensorNumpy(),
                # FIX: حذف transforms.Normalize
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif args.dataset == 'NYU':
            # --- FIX CRITICAL: اصلاح ساختار Normalization در حالت TRAIN ---
            self.train_transform = EnhancedCompose([
                RandomCropNumpy((args.height, args.width)),
                RandomHorizontalFlip(),
                [RandomColor(multiplier_range=(0.8, 1.2), brightness_mult_range=(0.75, 1.25)), None, None],
                ArrayToTensorNumpy(),
                # FIX: اینجا Normalization را به عنوان یک تابع مستقل قرار می‌دهیم!
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.test_transform = EnhancedCompose([
                ArrayToTensorNumpy(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, images, train=True):
        if train is True:
            return self.train_transform(images)
        else:
            return self.test_transform(images)