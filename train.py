import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
from tqdm import tqdm

# ایمپورت کردن ماژول‌های پروژه خودتان
from monoclip import MonoCLIP
from datasets.datasets_list import MyDataset

# --- تنظیمات اولیه ---
def get_args():
    parser = argparse.ArgumentParser(description='Train SAM-Enhanced DepthCLIP Adapter')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size') # با توجه به H100 می‌توانیم کمی بالا ببریم
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--data_path', type=str, default="/scratch/ram112/NYU_dataset", help='Path to dataset')
    parser.add_argument('--trainfile_nyu', type=str, default="./datasets/nyu_train.txt", help='Path to train split')
    parser.add_argument('--save_path', type=str, default="./checkpoints_trained", help='Where to save checkpoints')
    
    # آرگومان‌های مورد نیاز MyDataset (که در کد اصلی استفاده می‌شوند)
    parser.add_argument('--dataset', type=str, default='NYU')
    parser.add_argument('--use_dense_depth', action='store_true', default=True)
    parser.add_argument('--max_depth', type=float, default=10.0)
    parser.add_argument('--height', type=int, default=416)
    parser.add_argument('--width', type=int, default=544)
    parser.add_argument('--trainfile_kitti', type=str, default="")
    parser.add_argument('--testfile_kitti', type=str, default="")
    parser.add_argument('--testfile_nyu', type=str, default="") # فقط برای جلوگیری از ارور
    
    return parser.parse_args()

# --- تابع زیان (Loss Function) ---
class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        
    def forward(self, pred, target):
        # فقط پیکسل‌هایی که در GT مقدار دارند (Valid Pixels) را مقایسه کن
        mask = target > 0.001
        if mask.sum() == 0:
            return torch.tensor(0.0).to(pred.device).requires_grad_(True)
            
        diff = torch.abs(pred[mask] - target[mask])
        loss = torch.mean(diff)
        return loss

def main():
    args = get_args()
    
    # 1. ساخت پوشه چک‌پوینت
    os.makedirs(args.save_path, exist_ok=True)
    
    print("==== Initializing Model ====")
    # لود مدل (شامل SAM و CLIP)
    model = MonoCLIP()
    
    # 2. FREEZE کردن مدل (بسیار مهم)
    print("==== Freezing Backbones ====")
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. باز کردن قفل لایه‌های آداپتور برای آموزش
    # ما می‌خواهیم FCLayer و لایه Projection یاد بگیرند
    print("==== Unfreezing Adapters ====")
    trainable_params = []
    
    if hasattr(model, 'adapter'):
        for param in model.adapter.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        print("-> Adapter layer is trainable.")

    if hasattr(model, 'vis_to_text'):
        for param in model.vis_to_text.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        print("-> Projection layer (vis_to_text) is trainable.")

    # انتقال به GPU (اگر monoclip.py خودش منتقل نکند، اینجا اطمینان حاصل می‌کنیم)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        model.to(device)

    # 4. دیتالودر
    print("==== Loading Dataset ====")
    # نکته: فایل train لیست باید وجود داشته باشد. اگر ندارید، موقتاً از همان test list استفاده کنید.
    train_dataset = MyDataset(args, train=True) 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    print(f"Training on {len(train_dataset)} images.")

    # 5. تنظیمات آموزش
    criterion = MaskedL1Loss()
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    
    # --- Training Loop ---
    print("==== Starting Training ====")
    model.train() # حالت Train (برای Dropout و BatchNorm)
    # اما چون Backboneها فریز هستند، فقط آداپتور تغییر می‌کند.
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        # استفاده از tqdm برای نمایش پیشرفت
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, (rgb, gt, gt_dense) in enumerate(progress_bar):
            rgb = rgb.to(device)
            # استفاده از gt_dense اگر موجود باشد، وگرنه gt معمولی
            target = gt_dense.to(device) if args.use_dense_depth else gt.to(device)
            
            # Forward
            optimizer.zero_grad()
            pred_depth = model(rgb)
            
            # Loss
            loss = criterion(pred_depth, target)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # آپدیت نوار پیشرفت
            progress_bar.set_postfix({'Loss': loss.item()})
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch [{epoch+1}/{args.epochs}] Completed. Avg Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.1f}s")
        
        # ذخیره چک‌پوینت در هر اپوک
        save_name = os.path.join(args.save_path, f"sam_depthclip_epoch_{epoch+1}.pth")
        
        # ما فقط وزن‌های آداپتور را ذخیره می‌کنیم تا حجم کم باشد، یا کل مدل را
        # برای سادگی فعلاً کل state_dict را ذخیره می‌کنیم
        torch.save(model.state_dict(), save_name)
        print(f"Checkpoint saved: {save_name}")

if __name__ == "__main__":
    main()