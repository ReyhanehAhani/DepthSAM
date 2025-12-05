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

# --- تابع Argument Parser (با آرگومان‌های صحیح) ---
def get_args():
    parser = argparse.ArgumentParser(description='Train SAM-Enhanced DepthCLIP Adapter')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size') 
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--data_path', type=str, default="/scratch/ram112/NYU_dataset", help='Path to dataset')
    parser.add_argument('--trainfile_nyu', type=str, default="./datasets/my_test_list.txt", help='Path to train split') 
    parser.add_argument('--save_path', type=str, default="./checkpoints_trained", help='Where to save checkpoints')
    
    # آرگومان‌های مورد نیاز MyDataset
    parser.add_argument('--dataset', type=str, default='NYU')
    parser.add_argument('--use_dense_depth', action='store_true', default=True)
    parser.add_argument('--max_depth', type=float, default=10.0)
    parser.add_argument('--height', type=int, default=416)
    parser.add_argument('--width', type=int, default=544)
    parser.add_argument('--trainfile_kitti', type=str, default="")
    parser.add_argument('--testfile_kitti', type=str, default="")
    parser.add_argument('--testfile_nyu', type=str, default="") 
    
    return parser.parse_args()

# --- تابع زیان (Loss Function) ---
class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        
    def forward(self, pred, target):
        mask = target > 0.001
        
        if mask.sum() == 0:
            # FIX 1: تانسور خالی با requires_grad=True
            return torch.tensor(0.0, device=pred.device).requires_grad_(True)
            
        diff = torch.abs(pred[mask] - target[mask])
        
        loss = torch.mean(diff)
        
        # FIX 2 (CRITICAL): اطمینان از ردیابی گرادیان در تانسور Loss
        if not loss.requires_grad:
             loss.requires_grad_(True) 

        return loss

# --- Main Function ---
def main():
    args = get_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    print("==== Initializing Model ====")
    model = MonoCLIP()
    
    # 2. FREEZE کردن مدل
    print("==== Freezing Backbones ====")
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. باز کردن قفل لایه‌های آداپتور
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

    # 4. دیتالودر
    print("==== Loading Dataset ====")
    train_dataset = MyDataset(args, train=True) 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    print(f"Training on {len(train_dataset)} images.")

    # 5. تنظیمات آموزش
    criterion = MaskedL1Loss()
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    
    # --- Training Loop ---
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
            
            # --- FIX: Forward Pass با تضمین گرادیان و برش خروجی ---
            # استفاده از torch.set_grad_enabled(True) برای حل مشکل DP/Frozen Layer
            with torch.set_grad_enabled(True):
                pred_depth = model(rgb)
                
                # برش خروجی (Fix 1: Batch Size Mismatch)
                B = target.shape[0]
                pred_depth = pred_depth[:B]
                
                # Loss
                loss = criterion(pred_depth, target)
            # --------------------------------------------------------

            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            progress_bar.set_postfix({'Loss': loss.item()})
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch [{epoch+1}/{args.epochs}] Completed. Avg Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.1f}s")
        
        save_name = os.path.join(args.save_path, f"sam_depthclip_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_name)
        print(f"Checkpoint saved: {save_name}")

if __name__ == "__main__":
    main()