#!/usr/bin/env python3
import torch
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F

from monoclip import MonoCLIP  # DepthCLIP original model

def get_rgb_transform():
    """Resize + normalize to CLIP-friendly size 416×544 (13×17 patches with 32 stride)."""
    return transforms.Compose([
        transforms.Resize((416, 544)),  # <-- مهم: 416×544
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711))
    ])

def save_colormap(array_np, out_path, cmap='turbo'):
    vmin, vmax = np.percentile(array_np, 1), np.percentile(array_np, 99)
    norm = np.clip((array_np - vmin) / (vmax - vmin + 1e-8), 0, 1)
    plt.figure(figsize=(6, 4))
    plt.imshow(norm, cmap=cmap)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True, help='Path with rgb/ and depth/')
    p.add_argument('--out_dir', default='viz_outputs_depthclip_original', help='Where to save')
    p.add_argument('--num_samples', type=int, default=50)
    p.add_argument('--cmap', type=str, default='turbo')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load MonoCLIP (no DataParallel)
    print("[INFO] Creating MonoCLIP model...")
    model = MonoCLIP().to(device).eval()

    # Files
    rgb_files = sorted((Path(args.data_root) / "rgb").glob("*.jpg"))
    gt_files  = sorted((Path(args.data_root) / "depth").glob("*.png"))
    n = min(args.num_samples, len(rgb_files))
    print(f"[INFO] Running on {n} samples")

    tfm = get_rgb_transform()

    for idx in range(n):
        rgb_path = rgb_files[idx]
        gt_path  = gt_files[idx] if idx < len(gt_files) else None

        # Load & preprocess RGB
        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb_np  = np.array(rgb_img)
        x = tfm(rgb_img).unsqueeze(0).to(device)   # (1,3,416,544)

        with torch.no_grad():
            pred = model(x)  # DepthCLIP forward expects 416×544 → 13×17 patches (221)
            # upscale back to original RGB size for viz
            pred = F.interpolate(pred, size=rgb_np.shape[:2], mode='bilinear', align_corners=False)
            pred_np = pred.squeeze().cpu().numpy()

        # Save RGB
        Image.fromarray(rgb_np).save(out_dir / f"Sample_{idx+1:03d}.png")
        # Save Pred
        save_colormap(pred_np, out_dir / f"Depth_{idx+1:03d}.png", cmap=args.cmap)
        # Save GT (if available; NYU mm→m)
        if gt_path is not None:
            gt_np = np.array(Image.open(gt_path)).astype(np.float32) / 1000.0
            save_colormap(gt_np, out_dir / f"GT_{idx+1:03d}.png", cmap=args.cmap)

        if (idx+1) % 10 == 0:
            print(f"[{idx+1}/{n}] processed")

    print(f"[✓] Outputs saved in: {out_dir}")

if __name__ == "__main__":
    main()
