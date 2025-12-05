import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
from tqdm import tqdm
import sys

# ایمپورت کردن ماژول‌های پروژه
# فرض بر این است که فایل monoclip.py در همین پوشه است
from monoclip import MonoCLIP
from datasets.datasets_list import MyDataset

# --- 1. تنظیمات و آرگومان‌ها ---
def get_args():
    parser = argparse.ArgumentParser(description='Train SAM-Enhanced DepthCLIP Adapter')
    
    # تنظیمات آموزش
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size') 
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate') 
    
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
    
    return parser.parse_args()

# --- 2. توابع زیان پیشرفته (Composite Loss) ---

class GradientLoss(nn.Module):
    """محاسبه خطای گرادیان برای حفظ لبه‌ها و ساختار تصویر"""
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, pred, target, mask):
        # گرادیان افقی (X)
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        mask_dx = mask[:, :, :, :-1] & mask[:, :, :, 1:] 
        
        # گرادیان عمودی (Y)
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        mask_dy = mask[:, :, :-1, :] & mask[:, :, 1:, :] 

        loss_dx = torch.abs(pred_dx - target_dx)[mask_dx].mean() if mask_dx.sum() > 0 else 0.0
        loss_dy = torch.abs(pred_dy - target_dy)[mask_dy].mean() if mask_dy.sum() > 0 else 0.0

        return loss_dx + loss_dy

class CompositeLoss(nn.Module):
    """ترکیب L1 Loss و Gradient Loss - نسخه اصلاح شده"""
    def __init__(self, alpha=1.0, beta=0.5):
        super(CompositeLoss, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='none') 
        self.grad_loss = GradientLoss()
        self.alpha = alpha 
        self.beta = beta   

    def forward(self, pred, target):
        mask = target > 0.001
        
        if mask.sum() == 0:
            # برگرداندن صفر با گرادیان فعال برای جلوگیری از کرش
            return (pred * 0.0).sum()

        # محاسبه L1
        pixel_loss = self.l1_loss(pred, target)
        pixel_loss = pixel_loss[mask].mean()

        # محاسبه Gradient
        edge_loss = self.grad_loss(pred, target, mask)

        total_loss = (self.alpha * pixel_loss) + (self.beta * edge_loss)
        
        # --- SAFETY CHECK ---
        # اینجا دیگر دستی requires_grad را True نمی‌کنیم.
        # اگر True نباشد یعنی مدل قطع است و باید ارور بدهد.
        if not total_loss.requires_grad:
            raise RuntimeError("ERROR: Loss has no gradient flow! Check model connectivity.")

        return total_loss

# --- 3. بدنه اصلی (Main) ---
def main():
    args = get_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==== Running on {device} ====")
    
    print("==== Initializing Model ====")
    model = MonoCLIP()
    model.to(device)
    
    # --- مدیریت فریز/آن‌فریز ---
    print("==== Configuriing Trainable Parameters ====")
    # 1. همه چیز را فریز کن
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. فقط آداپتورها را باز کن
    trainable_params = []
    
    # باز کردن Adapter (لایه Residual)
    if hasattr(model, 'adapter'):
        for name, param in model.adapter.named_parameters():
            param.requires_grad = True
            trainable_params.append(param)
        print("-> Adapter layer UNFROZEN.")
    else:
        print("❌ CRITICAL WARNING: 'adapter' not found in model!")

    # باز کردن لایه پروجکشن (اگر Identity نباشد)
    if hasattr(model, 'vis_to_text') and not isinstance(model.vis_to_text, nn.Identity):
        for name, param in model.vis_to_text.named_parameters():
            param.requires_grad = True
            trainable_params.append(param)
        print("-> Projection layer (vis_to_text) UNFROZEN.")

    # چک نهایی تعداد پارامترهای قابل آموزش
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"==== Total Trainable Parameters: {num_trainable} ====")
    if num_trainable == 0:
        print("❌ ERROR: No parameters to train! Exiting.")
        sys.exit(1)

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

    # تنظیمات
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
            
            # پاک کردن گرادیان‌های قبلی
            optimizer.zero_grad()
            
            # Forward Pass
            pred_depth = model(rgb)
            
            # اطمینان از تطابق سایز (برش در صورت ناهماهنگی Batch آخر)
            if pred_depth.shape[0] != target.shape[0]:
                pred_depth = pred_depth[:target.shape[0]]
            
            loss = criterion(pred_depth, target)

            # Backward Pass
            loss.backward()
            
            # Gradient Clipping (برای پایداری)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(train_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        
        print(f"\nEpoch [{epoch+1}/{args.epochs}] Done. Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Time: {time.time() - start_time:.1f}s")
        
        # ذخیره مدل
        save_name = os.path.join(args.save_path, f"sam_depthclip_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_name)
        print(f"Checkpoint saved: {save_name}")

if __name__ == "__main__":
    main()