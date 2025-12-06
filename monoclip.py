import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import clip

# --- تلاش برای ایمپورت SAM 3 ---
try:
    from sam3.model_builder import build_sam3_image_model
    # print("✅ SAM 3 library imported successfully.")
except ImportError as e:
    print(f"❌ Error importing SAM 3: {e}")
    sys.exit(1)

# --- تنظیمات دستگاه ---
LOAD_DEVICE = 'cpu' # لود اولیه روی CPU برای جلوگیری از OOM
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- تنظیمات مسیر و مدل ---
SAM3_CHECKPOINT = "/home/ram112/projects/def-jieliang/ram112/checkpoints/sam3_large.pth"

# تنظیمات DepthCLIP
depth_templates = ['This {} is {}']
obj_classes = ['object']
depth_classes = ['giant', 'extremely close', 'close', 'not in distance', 'a little remote', 'far', 'unseen']
bin_list = [1.00, 1.50, 2.00, 2.25, 2.50, 2.75, 3.00]
temperature = 0.1

class SAM3Encoder(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        print(f"Loading SAM 3 Image Model from: {checkpoint_path}")
        
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            device=LOAD_DEVICE,  
            eval_mode=True,
            enable_segmentation=False,
            enable_inst_interactivity=False
        )
        
        if DEVICE == 'cuda':
            self.model.to(DEVICE)

        if hasattr(self.model.backbone, 'visual'):
            self.image_encoder = self.model.backbone.visual
        elif hasattr(self.model.backbone, 'trunk'):
            self.image_encoder = self.model.backbone.trunk
        else:
            self.image_encoder = self.model.backbone
        
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        batch_size = x.shape[0]
        dummy_captions = [''] * batch_size
        
        if x.shape[-2:] != (1008, 1008):
            x_in = F.interpolate(x, size=(1008, 1008), mode='bilinear', align_corners=False)
        else:
            x_in = x

        try:
            features = self.image_encoder(x_in, captions=dummy_captions)
        except TypeError:
            features = self.image_encoder(x_in)
        
        if isinstance(features, dict):
            last_key = list(features.keys())[-1]
            return features[last_key]
        elif isinstance(features, (list, tuple)):
            return features[-1]
            
        return features

def get_text_features(clip_model, depth_classes, obj_classes, templates):
    zeroshot_weights = []
    with torch.no_grad():
        for depth in depth_classes:
            for obj in obj_classes:
                texts = [template.format(obj, depth) for template in templates]
                texts = clip.tokenize(texts).to(DEVICE)
                class_embeddings = clip_model.encode_text(texts).to(torch.float32) 
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
    
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(DEVICE).to(torch.float32)
    return zeroshot_weights

class DepthAdapterCNN(nn.Module):
    """
    آداپتور جدید مبتنی بر CNN برای درک ویژگی‌های مکانی (Spatial Features).
    جایگزین FCLayer قدیمی شد.
    """
    def __init__(self, c_in, reduction=4):
        super(DepthAdapterCNN, self).__init__()
        
        reduced_dim = max(c_in // reduction, 64) # جلوگیری از خیلی کوچک شدن ابعاد
        
        self.adapter = nn.Sequential(
            # 1. کاهش ابعاد (1x1 Conv)
            nn.Conv2d(c_in, reduced_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_dim),
            nn.ReLU(inplace=True),
            
            # 2. درک مکانی (3x3 Conv) -> اینجاست که عمق فهمیده میشه
            nn.Conv2d(reduced_dim, reduced_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduced_dim),
            nn.ReLU(inplace=True),
            
            # 3. بازیابی ابعاد (1x1 Conv)
            nn.Conv2d(reduced_dim, c_in, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_in)
            # نکته: ReLU آخر رو برمیداریم تا بتونه مقادیر منفی رو هم در Residual اصلاح کنه
        )
        
        # مقداردهی اولیه وزن‌ها برای شروع نرم (نزدیک به صفر)
        # این باعث میشه در شروع کار، تاثیر آداپتور کم باشه و مدل منفجر نشه
        nn.init.constant_(self.adapter[-1].weight, 0) 

    def forward(self, x):
        # ورودی معمولاً به شکل (B, H, W, C) است
        # کانولوشن نیاز به (B, C, H, W) دارد
        
        x = x.permute(0, 3, 1, 2) # تبدیل به فرمت کانال-اول
        x = self.adapter(x)
        x = x.permute(0, 2, 3, 1) # برگرداندن به فرمت کانال-آخر
        
        return x

class MonoCLIP(nn.Module):
    def __init__(self):
        super(MonoCLIP, self).__init__()
        self.bins = len(depth_classes)

        print("Loading CLIP (RN50) for text encoding...")
        self.clip_model, _ = clip.load("RN50", device=LOAD_DEVICE)
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        if DEVICE == 'cuda':
            self.clip_model.to(DEVICE)

        self.text_f = get_text_features(self.clip_model, depth_classes, obj_classes, depth_templates)
        self.text_dim = 1024

        self.sam_encoder = SAM3Encoder(SAM3_CHECKPOINT)
        
        # چک کردن سایز خروجی
        dummy = torch.randn(1, 3, 1008, 1008).to(DEVICE)
        with torch.no_grad():
            out = self.sam_encoder(dummy)
        
        # اگر خروجی 3 بعدی بود (Batch, Seq, Dim)، باید سایز تصویر رو حدس بزنیم
        if out.dim() == 3:
            self.visual_dim = out.shape[-1]
            # معمولاً SAM3 خروجی 64x64 میده اگر ورودی 1024 باشه (Patch Size 16)
            self.spatial_size = int(out.shape[1] ** 0.5) 
        else:
            self.visual_dim = out.shape[1] # اگر (B, C, H, W) باشه
            
        print(f"✅ SAM 3 Output Shape: {out.shape}")
        print(f"✅ Visual Dimension: {self.visual_dim}")

        # --- استفاده از آداپتور CNN جدید ---
        self.adapter = DepthAdapterCNN(self.visual_dim).to(DEVICE)
        
        if self.visual_dim != self.text_dim:
            self.vis_to_text = nn.Linear(self.visual_dim, self.text_dim, bias=False).to(DEVICE)
        else:
            self.vis_to_text = nn.Identity()

    def forward(self, x):
        # 1. فیچرها از SAM
        img_f = self.sam_encoder(x)
        img_f = img_f.to(torch.float32)

        # 2. استاندارد سازی شکل تنسور به (B, H, W, C)
        if img_f.dim() == 3: # اگر (B, Seq, C) بود
            B, Seq, C = img_f.shape
            H = W = int(Seq ** 0.5)
            img_f = img_f.view(B, H, W, C)
        elif img_f.dim() == 4 and img_f.shape[1] == self.visual_dim: # اگر (B, C, H, W) بود
            img_f = img_f.permute(0, 2, 3, 1)

        # 3. نرمال‌سازی اولیه
        img_f = img_f / (img_f.norm(dim=-1, keepdim=True) + 1e-6)

        # -----------------------------------------------------------
        # 🔥 CNN ADAPTER (Residual)
        # -----------------------------------------------------------
        img_f = img_f + self.adapter(img_f)
        
        # 4. پروجکشن به فضای متن
        img_f = self.vis_to_text(img_f)
        
        # 5. محاسبه عمق
        depth_logits = 100. * img_f @ self.text_f
        
        # تبدیل به (B, Classes, H, W) برای Softmax
        depth_logits = depth_logits.permute(0, 3, 1, 2)
        
        depth_logits /= temperature
        depth_probs = F.softmax(depth_logits, dim=1)
        
        bin_tensor = torch.tensor(bin_list).to(depth_probs.device)
        depth_map = (depth_probs * bin_tensor.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        
        if depth_map.shape[-2:] != x.shape[-2:]:
            depth_map = F.interpolate(depth_map, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return depth_map