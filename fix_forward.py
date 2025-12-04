import os

file_path = "/home/ram112/projects/def-jieliang/ram112/All_DEPTHCLIP/DepthCLIP/DepthCLIP_code/monoclip.py"

print(f"Patching forward method in {file_path}...")

with open(file_path, 'r') as f:
    lines = f.readlines()

new_lines = []
found = False

for line in lines:
    # دنبال خطی می‌گردیم که فراخوانی انکودر انجام می‌شود
    # قبلاً: features = self.image_encoder(x)
    if "features = self.image_encoder(x)" in line:
        found = True
        indent = line[:line.find("features")] # حفظ تورفتگی
        
        print("-> Found target line. Injecting dummy captions...")
        
        # کد جایگزین: ساخت کپشن خالی و ارسال آن
        replacement = [
            f"{indent}# --- FIX BY GEMINI (Pass Dummy Captions) ---\n",
            f"{indent}batch_size = x.shape[0]\n",
            f"{indent}# مدل SAM 3 VL Backbone نیاز به ورودی متن دارد.\n",
            f"{indent}# ما یک لیست متن خالی یا عمومی می دهیم.\n",
            f"{indent}dummy_captions = [''] * batch_size\n",
            f"{indent}try:\n",
            f"{indent}    features = self.image_encoder(x, captions=dummy_captions)\n",
            f"{indent}except TypeError:\n",
            f"{indent}    # محض اطمینان اگر آرگومان نامش چیز دیگری بود\n",
            f"{indent}    print('DEBUG: passing captions failed, trying input_ids or raw x')\n",
            f"{indent}    features = self.image_encoder(x)\n",
            f"{indent}# --- FIX END ---\n"
        ]
        new_lines.extend(replacement)
    else:
        new_lines.append(line)

if found:
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    print("✅ Successfully patched monoclip.py forward method!")
else:
    print("⚠️ Warning: Could not find 'features = self.image_encoder(x)' line.")
    print("Please check if the file content matches expectations.")

