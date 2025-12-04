import h5py
import numpy as np
import cv2
import os
from tqdm import tqdm

# تنظیم مسیرها روی فضای scratch
mat_path = '/scratch/ram112/NYU_dataset/nyu_depth_v2_labeled.mat'
output_dir = '/scratch/ram112/NYU_dataset'

if not os.path.exists(mat_path):
    print(f"Error: File not found at {mat_path}")
    exit(1)

print(f"Loading {mat_path}...")
f = h5py.File(mat_path, 'r')

# خواندن متغیرها
images = f['images']
depths = f['depths']
num_images = images.shape[0]

print(f"Found {num_images} images. Extracting to {output_dir}...")

for i in tqdm(range(num_images)):
    # 1. استخراج و اصلاح تصویر RGB
    img = np.array(images[i])
    # تبدیل ابعاد از فرمت متلب (C, W, H) به (H, W, C)
    img = np.transpose(img, (2, 1, 0)) 
    # تبدیل رنگ از RGB به BGR (استاندارد OpenCV)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 2. استخراج و اصلاح نقشه عمق
    depth = np.array(depths[i])
    # تبدیل ابعاد از (W, H) به (H, W)
    depth = np.transpose(depth, (1, 0))
    
    # نام‌گذاری فایل‌ها طبق فرمت استاندارد لیست‌های متنی
    # (مثلاً rgb_00000.jpg و sync_depth_00000.png)
    img_name = f"rgb_{i:05d}.jpg"
    depth_name = f"sync_depth_{i:05d}.png"
    
    # 3. ذخیره‌سازی
    cv2.imwrite(os.path.join(output_dir, img_name), img)
    
    # عمق را در مقیاس میلی‌متر (16 بیتی) ذخیره می‌کنیم
    # (مقادیر اصلی متر هستند، ضرب در 1000 می‌شوند)
    depth_uint16 = (depth * 1000).astype(np.uint16)
    cv2.imwrite(os.path.join(output_dir, depth_name), depth_uint16)

print("✅ Extraction Complete!")
