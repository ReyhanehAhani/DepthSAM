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
# برای جلوگیری از پر شدن حافظه GPU هنگام لود اولیه، مدل را روی CPU لود می‌کنیم
LOAD_DEVICE = 'cpu'
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
        
        # 1. لود مدل روی CPU برای مدیریت حافظه
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            device=LOAD_DEVICE,  
            eval_mode=True,
            enable_segmentation=False,
            enable_inst_interactivity=False
        )
        
        # 2. انتقال به GPU اگر موجود باشد
        if DEVICE == 'cuda':
            self.model.to(DEVICE)

        # 3. پیدا کردن اینکودر تصویر (Smart Backbone Detection)
        if hasattr(self.model.backbone, 'visual'):
            self.image_encoder = self.model.backbone.visual
        elif hasattr(self.model.backbone, 'trunk'):
            self.image_encoder = self.model.backbone.trunk
        else:
            self.image_encoder = self.model.backbone
        
        # فریز کردن کامل SAM
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        ورودی: تنسور تصویر (B, 3, H, W)
        """
        batch_size = x.shape[0]
        dummy_captions = [''] * batch_size
        
        # تغییر سایز برای RoPE (معمولاً SAM روی ۱۰۲۴ یا ۱۰۰۸ کار می‌کند)
        if x.shape[-2:] != (1008, 1008):
            x_in = F.interpolate(x, size=(1008, 1008), mode='bilinear', align_corners=False)
        else:
            x_in = x

        # هندل کردن ورودی‌های مختلف مدل (با یا بدون caption)
        try:
            features = self.image_encoder(x_in, captions=dummy_captions)
        except TypeError:
            features = self.image_encoder(x_in)
        
        # استخراج ویژگی نهایی از خروجی‌های چندگانه
        if isinstance(features, dict):
            last_key = list(features.keys())[-1]
            return features[last_key]
        elif isinstance(features, (list, tuple)):
            return features[-1]
            
        return features

def get_text_features(clip_model, depth_classes, obj_classes, templates):
    """
    تولید ویژگی‌های متنی CLIP برای کلاس‌های عمق.
    """
    zeroshot_weights = []
    with torch.no_grad(): # محاسبه فقط یکبار انجام می‌شود
        for depth in depth_classes:
            for obj in obj_classes:
                texts = [template.format(obj, depth) for template in templates]
                texts = clip.tokenize(texts).to(DEVICE)
                
                # استفاده از float32 برای دقت بالا
                class_embeddings = clip_model.encode_text(texts).to(torch.float32) 
                
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
    
    # استک کردن و جدا کردن از گراف محاسباتی (Detached)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(DEVICE).to(torch.float32)
    return zeroshot_weights

class FCLayer(nn.Module):
    """
    لایه آداپتور ساده (MLP)
    """
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
        
        # فریز کردن CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        if DEVICE == 'cuda':
            self.clip_model.to(DEVICE)

        # محاسبه ویژگی‌های متنی (فیکس شده)
        self.text_f = get_text_features(self.clip_model, depth_classes, obj_classes, depth_templates)
        self.text_dim = 1024

        # لود SAM Encoder
        self.sam_encoder = SAM3Encoder(SAM3_CHECKPOINT)
        
        # بررسی ابعاد خروجی SAM برای ساخت آداپتور
        dummy = torch.randn(1, 3, 1008, 1008).to(DEVICE)
        with torch.no_grad():
            out = self.sam_encoder(dummy)
        
        self.visual_dim = out.shape[-1] # فرض بر این است که کانال در دایمنشن آخر است
        print(f"✅ SAM 3 Output Shape: {out.shape}")
        print(f"✅ Visual Dimension: {self.visual_dim}")

        # --- تعریف لایه‌های قابل آموزش ---
        self.adapter = FCLayer(self.visual_dim).to(DEVICE)
        
        if self.visual_dim != self.text_dim:
            self.vis_to_text = nn.Linear(self.visual_dim, self.text_dim, bias=False).to(DEVICE)
        else:
            self.vis_to_text = nn.Identity()

    def forward(self, x):
        # 1. استخراج ویژگی از SAM (فریز شده)
        img_f = self.sam_encoder(x)
        
        # 2. تبدیل به float32 برای پایداری محاسبات و جلوگیری از NaN
        img_f = img_f.to(torch.float32)

        # -----------------------------------------------------------
        # 🔥 FIX CRITICAL: اعمال آداپتور به صورت Residual
        # این خط باعث می‌شود گرادیان جریان پیدا کند و لاس کم شود
        # -----------------------------------------------------------
        img_f = img_f + self.adapter(img_f)
        
        # 3. نرمال‌سازی و تغییر شکل (Reshape)
        # اگر خروجی 3 بعدی است (B, Seq, Dim) تبدیل به فرمت تصویر (B, H, W, Dim)
        if img_f.dim() == 3:
            # فرض بر این است که spatial dimension فشرده شده است، اینجا باز می‌کنیم
            # نکته: اگر لاجیک خاصی برای SAM3 دارید اینجا چک کنید. 
            # در کد قبلی شما اینطور بود:
            img_f = img_f.transpose(1, 2)  # (B, Dim, Seq)
            img_f = img_f.unsqueeze(-1)    # (B, Dim, Seq, 1) - تبدیل موقت به 4D
        
        img_f = img_f / (img_f.norm(dim=1, keepdim=True) + 1e-6)

        # تبدیل به فرمت (B, H, W, C) برای عبور از لایه‌های خطی
        if img_f.shape[1] == self.visual_dim: # اگر کانال در دایمنشن 1 است
             img_f = img_f.permute(0, 2, 3, 1) 
        
        # 4. پروجکشن به فضای متن
        img_f = self.vis_to_text(img_f)
        
        # 5. محاسبه امتیاز عمق (شباهت تصویر و متن)
        depth_logits = 100. * img_f @ self.text_f
        
        # آماده‌سازی برای Softmax (B, Classes, H, W)
        depth_logits = depth_logits.permute(0, 3, 1, 2)
        
        depth_logits /= temperature
        depth_probs = F.softmax(depth_logits, dim=1)
        
        # 6. محاسبه نقشه عمق نهایی (Weighted Sum)
        bin_tensor = torch.tensor(bin_list).to(depth_probs.device)
        depth_map = (depth_probs * bin_tensor.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        
        # 7. بازگرداندن به سایز اصلی تصویر ورودی
        if depth_map.shape[-2:] != x.shape[-2:]:
            depth_map = F.interpolate(depth_map, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return depth_map