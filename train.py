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

# --- 1. تنظیمات و آرگومان‌ها ---
def get_args():
    parser = argparse.ArgumentParser(description='Train SAM-Enhanced DepthCLIP Adapter')
    
    # تنظیمات آموزش
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size') 
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate') # شروع با نرخ بالا
    
    # مسیرها
    parser.add_argument('--data_path', type=str, default="/scratch/ram112/NYU_dataset", help='Path to dataset')
    parser.add_argument('--trainfile_nyu', type=str, default="./datasets/my_test_list.txt", help='Path to train split') 
    parser.add_argument('--save_path', type=str, default="./checkpoints_trained", help='Where to save checkpoints')
    
    # تنظیمات مدل و دیتاست
    parser.add_argument('--dataset', type=str, default='NYU')
    parser.add_argument('--use_dense_depth', action='store_true', default=True)
    parser.add_argument('--max_depth', type=float, default=10.0)
    parser.add_argument('--height', type=int, default=416)
    parser.add_argument('--width', type=int, default=544)
    
    # آرگومان‌های اضافی
    parser.add_argument('--trainfile_kitti', type=str, default="")
    parser.add_argument('--testfile_kitti', type=str, default="")
    parser.add_argument('--testfile_nyu', type=str, default="") 
    
    return parser.parse_args()

# --- 2. توابع زیان پیشرفته (Composite Loss) ---

class GradientLoss(nn.Module):
    """محاسبه خطای گرادیان برای حفظ لبه‌ها و ساختار تصویر"""
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, pred, target, mask):
        # محاسبه گرادیان (تغییرات) در جهت X و Y
        # تانسورها به شکل (B, C, H, W) هستند
        
        # گرادیان افقی (X)
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        mask_dx = mask[:, :, :, :-1] & mask[:, :, :, 1:] # ماسک مشترک
        
        # گرادیان عمودی (Y)
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        mask_dy = mask[:, :, :-1, :] & mask[:, :, 1:, :] # ماسک مشترک

        # محاسبه خطا فقط در جاهایی که ماسک معتبر است
        if mask_dx.sum() > 0:
            loss_dx = torch.abs(pred_dx - target_dx)[mask_dx].mean()
        else:
            loss_dx = 0

        if mask_dy.sum() > 0:
            loss_dy = torch.abs(pred_dy - target_dy)[mask_dy].mean()
        else:
            loss_dy = 0

        return loss_dx + loss_dy

class CompositeLoss(nn.Module):
    """ترکیب L1 Loss و Gradient Loss"""
    def __init__(self, alpha=1.0, beta=0.5):
        super(CompositeLoss, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='none') 
        self.grad_loss = GradientLoss()
        self.alpha = alpha # وزن L1 (دقت کلی)
        self.beta = beta   # وزن Gradient (دقت لبه‌ها)

    def forward(self, pred, target):
        # 1. ساخت ماسک معتبر
        mask = target > 0.001
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device).requires_grad_(True)

        # 2. محاسبه L1 Loss (فقط روی پیکسل‌های معتبر)
        pixel_loss = self.l1_loss(pred, target)
        pixel_loss = pixel_loss[mask].mean()

        # 3. محاسبه Gradient Loss (برای تیز کردن لبه‌ها)
        edge_loss = self.grad_loss(pred, target, mask)

        # 4. ترکیب نهایی
        total_loss = (self.alpha * pixel_loss) + (self.beta * edge_loss)
        
        # FIX: اطمینان از جریان گرادیان
        if not total_loss.requires_grad:
             total_loss.requires_grad_(True)

        return total_loss

# --- 3. بدنه اصلی (Main) ---
def main():
    args = get_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    print("==== Initializing Model ====")
    model = MonoCLIP()
    
    # فریز کردن بخش‌های سنگین
    print("==== Freezing Backbones ====")
    for param in model.parameters():
        param.requires_grad = False
        
    # باز کردن قفل بخش‌های آداپتور
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        model.to(device)

    # دیتالودر
    print("==== Loading Dataset ====")
    train_dataset = MyDataset(args, train=True) 
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    print(f"Training on {len(train_dataset)} images.")

    # تنظیمات بهینه‌ساز و لاس ترکیبی
    # استفاده از CompositeLoss به جای MaskedL1Loss ساده
    criterion = CompositeLoss(alpha=1.0, beta=0.5)
    
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    print("==== Starting Training ====")
    model.train() 
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, (rgb, gt, gt_dense) in enumerate(progress_bar):
            rgb = rgb.to(device)
            target = gt_dense.to(device) if args.use_dense_depth else gt.to(device)
            
            optimizer.zero_grad()
            
            # --- FIX: جریان گرادیان امن ---
            with torch.set_grad_enabled(True):
                pred_depth = model(rgb)
                
                # برش خروجی برای تطابق با Batch Size
                B = target.shape[0]
                pred_depth = pred_depth[:B]
                
                loss = criterion(pred_depth, target)

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})
            
        avg_loss = epoch_loss / len(train_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        
        print(f"\nEpoch [{epoch+1}/{args.epochs}] Completed. Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Time: {time.time() - start_time:.1f}s")
        
        save_name = os.path.join(args.save_path, f"sam_depthclip_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_name)
        print(f"Checkpoint saved: {save_name}")

if __name__ == "__main__":
    main()