import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import clip

# --- FIX 1: تعریف دستگاه لود موقت ---
LOAD_DEVICE = 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"INFO: Running on device: {DEVICE} (Load device: {LOAD_DEVICE})")

# --- تنظیمات مسیر و مدل ---
SAM3_CHECKPOINT = "/home/ram112/projects/def-jieliang/ram112/checkpoints/sam3_large.pth"
# ... (بقیه تنظیمات ثابت) ...
depth_templates = ['This {} is {}']
obj_classes = ['object']
depth_classes = ['giant', 'extremely close', 'close', 'not in distance', 'a little remote', 'far', 'unseen']
bin_list = [1.00, 1.50, 2.00, 2.25, 2.50, 2.75, 3.00]
temperature = 0.1

# ... (SAM3Encoder و بقیه کدها بدون تغییر) ...

class SAM3Encoder(nn.Module):
# ... (بخش __init__ و forward بدون تغییر) ...
    def __init__(self, checkpoint_path):
        super().__init__()
        print(f"Loading SAM 3 Image Model from: {checkpoint_path}")
        
        # FIX 2: لود روی CPU 
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            device=LOAD_DEVICE,  
            eval_mode=True,
            enable_segmentation=False,
            enable_inst_interactivity=False
        )
        # FIX 3: انتقال مدل به GPU بعد از لود 
        if DEVICE == 'cuda':
            self.model.to(DEVICE)
        
        # ... (بخش Backbone Detection بدون تغییر) ...
        # ... (freezing remains the same) ...

    def forward(self, x):
        # ... (forward pass logic remains the same) ...
        # ... (slicing and unsqueeze logic remains the same) ...
        return features

def get_text_features(clip_model, depth_classes, obj_classes, templates):
    # توجه: با فرض اینکه شما WITH torch.no_grad(): را از monoclip.py حذف کرده‌اید!
    with torch.no_grad(): 
        zeroshot_weights = []
        for depth in depth_classes:
            for obj in obj_classes:
                texts = [template.format(obj, depth) for template in templates]
                texts = clip.tokenize(texts).to(DEVICE)
                
                # FIX 7 (CRITICAL): اطمینان از استفاده از float32 برای پایداری ضرب ماتریسی
                class_embeddings = clip_model.encode_text(texts).to(torch.float32) 
                
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(DEVICE).to(torch.float32)
    return zeroshot_weights

class FCLayer(nn.Module):
# ... (کلاس FCLayer بدون تغییر) ...
    def __init__(self, c_in, reduction=4):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fc(x)

class MonoCLIP(nn.Module):
    def __init__(self):
        super(MonoCLIP, self).__init__()
        self.bins = len(depth_classes)

        print("Loading CLIP (RN50) for text encoding...")
        self.clip_model, _ = clip.load("RN50", device=LOAD_DEVICE)
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # انتقال CLIP به GPU
        if DEVICE == 'cuda':
            self.clip_model.to(DEVICE)

        self.text_f = get_text_features(self.clip_model, depth_classes, obj_classes, depth_templates)
        self.text_dim = 1024

        self.sam_encoder = SAM3Encoder(SAM3_CHECKPOINT)
        
        # --- اینجا WITH torch.no_grad(): را حذف کرده‌اید ---
        dummy = torch.randn(1, 3, 1008, 1008).to(DEVICE)
        out = self.sam_encoder(dummy)
        
        self.visual_dim = out.shape[2] 
        print(f"✅ SAM 3 Output Shape: {out.shape}")
        print(f"✅ Visual Dimension: {self.visual_dim}")

        self.adapter = FCLayer(self.visual_dim).to(DEVICE)
        
        if self.visual_dim != self.text_dim:
            self.vis_to_text = nn.Linear(self.visual_dim, self.text_dim, bias=False).to(DEVICE)
            print(f"ℹ️ Projection layer added: {self.visual_dim} -> {self.text_dim}")
        else:
            self.vis_to_text = nn.Identity()

    def forward(self, x):
        img_f = self.sam_encoder(x)
        
        # --- FIX 6: Reshape 3D features to 4D dense feature map (1x1 spatial) ---
        if img_f.dim() == 3:
            img_f = img_f.transpose(1, 2)  
            img_f = img_f.unsqueeze(-1)    
        # --- END FIX 6 ---
        
        # FIX 8: اطمینان از float32 بودن تصویر برای ضرب ماتریسی
        img_f = img_f.to(torch.float32)
        
        img_f = img_f / (img_f.norm(dim=1, keepdim=True) + 1e-6)

        img_f = img_f.permute(0, 2, 3, 1) 
        
        img_f = self.vis_to_text(img_f)
        
        depth_logits = 100. * img_f @ self.text_f
        
        depth_logits = depth_logits.permute(0, 3, 1, 2)
        
        depth_logits /= temperature
        depth_probs = F.softmax(depth_logits, dim=1)
        
        # ... (بقیه کدها بدون تغییر) ...
        bin_tensor = torch.tensor(bin_list).to(depth_probs.device)
        depth_map = (depth_probs * bin_tensor.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        
        if depth_map.shape[-2:] != x.shape[-2:]:
            depth_map = F.interpolate(depth_map, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return depth_map