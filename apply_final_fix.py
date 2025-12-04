import os

file_path = "/home/ram112/projects/def-jieliang/ram112/All_DEPTHCLIP/DepthCLIP/DepthCLIP_code/monoclip.py"

print(f"Applying FINAL RESIZE PATCH to {file_path}...")

with open(file_path, 'r') as f:
    content = f.read()

# این دقیقاً همان بخشی است که در فایل فعلی شما وجود دارد (من از فایلی که فرستادی کپی کردم)
old_block_signature = """        # --- FIX BY GEMINI (Pass Dummy Captions) ---
        batch_size = x.shape[0]
        # مدل SAM 3 VL Backbone نیاز به ورودی متن دارد.
        # ما یک لیست متن خالی یا عمومی می دهیم.
        dummy_captions = [''] * batch_size
        try:
            features = self.image_encoder(x, captions=dummy_captions)
        except TypeError:
            # محض اطمینان اگر آرگومان نامش چیز دیگری بود
            print('DEBUG: passing captions failed, trying input_ids or raw x')
            features = self.image_encoder(x)
        # --- FIX END ---"""

# این بخش جدید است که هم عکس را بزرگ می‌کند و هم کپشن می‌دهد
new_block = """        # --- FIX BY GEMINI V2 (Resize 1024 + Dummy Captions) ---
        batch_size = x.shape[0]
        dummy_captions = [''] * batch_size
        
        # SAM 3 requires 1024x1024 input
        if x.shape[-2:] != (1024, 1024):
            # Upsample input to 1024x1024
            x_in = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
        else:
            x_in = x

        try:
            features = self.image_encoder(x_in, captions=dummy_captions)
        except TypeError:
            features = self.image_encoder(x_in)
        # --- FIX END ---"""

# جایگزینی دقیق متن
if old_block_signature in content:
    new_content = content.replace(old_block_signature, new_block)
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    print("✅ Success! The file has been updated with Resize logic (1024x1024).")
else:
    print("⚠️ Warning: Could not find the exact old code block.")
    print("Trying a smarter replacement based on markers...")
    
    # روش جایگزین هوشمند (اگر فاصله‌ها کمی فرق داشت)
    import re
    pattern = re.compile(r'# --- FIX BY GEMINI \(Pass Dummy Captions\).*?# --- FIX END ---', re.DOTALL)
    if pattern.search(content):
        new_content = pattern.sub(new_block, content)
        with open(file_path, 'w') as f:
            f.write(new_content)
        print("✅ Success! Updated using smart regex match.")
    else:
        print("❌ Error: Could not verify patch location. Please check monoclip.py manually.")

