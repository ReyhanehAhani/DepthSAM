import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import clip

# --- FIX 1: تعریف دستگاه لود موقت ---
# ما مدل را موقتاً روی CPU لود می‌کنیم تا از کرش سخت‌کد شده SAM3 جلوگیری کنیم.
LOAD_DEVICE = 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"INFO: Running on device: {DEVICE} (Load device: {LOAD_DEVICE})")

# --- تنظیمات مسیر و مدل ---
SAM3_CHECKPOINT = "/home/ram112/projects/def-jieliang/ram112/checkpoints/sam3_large.pth"

# تنظیمات DepthCLIP
depth_templates = ['This {} is {}']
obj_classes = ['object']
depth_classes = ['giant', 'extremely close', 'close', 'not in distance', 'a little remote', 'far', 'unseen']
bin_list = [1.00, 1.50, 2.00, 2.25, 2.50, 2.75, 3.00]
temperature = 0.1

# --- تلاش برای ایمپورت SAM 3 ---
try:
    from sam3.model_builder import build_sam3_image_model
    print("✅ SAM 3 library imported successfully.")
except ImportError as e:
    print(f"❌ Error importing SAM 3: {e}")
    print("Please ensure 'sam3' is installed or added to PYTHONPATH.")
    sys.exit(1)

class SAM3Encoder(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        print(f"Loading SAM 3 Image Model from: {checkpoint_path}")
        
        # FIX 2: لود روی CPU (حیاتی برای رفع ارور CUDA unknown error)
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            device=LOAD_DEVICE,  # <--- این خط جلوی کرش را می‌گیرد
            eval_mode=True,
            enable_segmentation=False,
            enable_inst_interactivity=False
        )
        
        # --- FIX 3: انتقال مدل به GPU بعد از لود ---
        if DEVICE == 'cuda':
            self.model.to(DEVICE) # انتقال مدل به GPU بعد از عبور از منطقه خطر

        # --- FIX 4: Smart Backbone Detection ---
        print(f'DEBUG: SAM3 Backbone Type: {type(self.model.backbone)}')
        if hasattr(self.model.backbone, 'visual'):
            self.image_encoder = self.model.backbone.visual
        elif hasattr(self.model.backbone, 'trunk'):
            print('DEBUG: Found .trunk, using it as image_encoder')
            self.image_encoder = self.model.backbone.trunk
        else:
            print('DEBUG: No .visual or .trunk found. Using backbone itself as image_encoder')
            self.image_encoder = self.model.backbone
        # --- FIX END ---
        
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        ورودی: تنسور تصویر (B, 3, H, W)
        """
        # --- FIX 3 & 4: Resize 1008 + Dummy Captions ---
        batch_size = x.shape[0]
        dummy_captions = [''] * batch_size
        
        # FIX 3: تغییر سایز برای RoPE
        if x.shape[-2:] != (1008, 1008):
            x_in = F.interpolate(x, size=(1008, 1008), mode='bilinear', align_corners=False)
        else:
            x_in = x

        try:
            # FIX 4: ارسال کپشن خالی برای VLM
            features = self.image_encoder(x_in, captions=dummy_captions)
        except TypeError:
            features = self.image_encoder(x_in)
        # --- FIX END ---
        
        if isinstance(features, dict):
            last_key = list(features.keys())[-1]
            return features[last_key]
        elif isinstance(features, (list, tuple)):
            return features[-1]
            
        return features

def get_text_features(clip_model, depth_classes, obj_classes, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for depth in depth_classes:
            for obj in obj_classes:
                texts = [template.format(obj, depth) for template in templates]
                texts = clip.tokenize(texts).to(DEVICE)
                class_embeddings = clip_model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(DEVICE)
    return zeroshot_weights

class FCLayer(nn.Module):
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
        # FIX 4: لود روی CPU
        self.clip_model, _ = clip.load("RN50", device=LOAD_DEVICE)
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # انتقال CLIP به GPU
        if DEVICE == 'cuda':
            self.clip_model.to(DEVICE)

        self.text_f = get_text_features(self.clip_model, depth_classes, obj_classes, depth_templates)
        self.text_dim = 1024

        self.sam_encoder = SAM3Encoder(SAM3_CHECKPOINT)
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, 1008, 1008).to(DEVICE)
            out = self.sam_encoder(dummy)
            # FIX 5: استفاده از اندیس 2 برای بعد فیچر (1024)
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
            # Output is (B, 1, D). Convert to (B, D, 1, 1).
            img_f = img_f.transpose(1, 2)  # (B, D, 1)
            img_f = img_f.unsqueeze(-1)    # (B, D, 1, 1)
        # --- END FIX 6 ---
        
        img_f = img_f / (img_f.norm(dim=1, keepdim=True) + 1e-6)

        # تبدیل (B, C, Hf, Wf) به (B, Hf, Wf, C)
        img_f = img_f.permute(0, 2, 3, 1) 
        
        img_f = self.vis_to_text(img_f)
        
        depth_logits = 100. * img_f @ self.text_f
        
        depth_logits = depth_logits.permute(0, 3, 1, 2)
        
        depth_logits /= temperature
        depth_probs = F.softmax(depth_logits, dim=1)
        
        bin_tensor = torch.tensor(bin_list).to(depth_probs.device)
        depth_map = (depth_probs * bin_tensor.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        
        if depth_map.shape[-2:] != x.shape[-2:]:
            depth_map = F.interpolate(depth_map, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return depth_map