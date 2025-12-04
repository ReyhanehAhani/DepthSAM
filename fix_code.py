import os

file_path = "/home/ram112/projects/def-jieliang/ram112/All_DEPTHCLIP/DepthCLIP/DepthCLIP_code/monoclip.py"

print(f"reading {file_path}...")

with open(file_path, 'r') as f:
    lines = f.readlines()

new_lines = []
found = False

for line in lines:
    # دنبال خطی می‌گردیم که ارور داده است
    if "self.image_encoder = self.model.backbone.visual" in line:
        found = True
        indent = line[:line.find("self.image_encoder")] # حفظ تورفتگی (Indentation)
        
        # کد جایگزین: اول چاپ می‌کند چه ویژگی‌هایی دارد، بعد سعی می‌کند درستش را انتخاب کند
        replacement = [
            f"{indent}# --- FIX BY GEMINI START ---\n",
            f"{indent}print(f'DEBUG: SAM3 Backbone Type: {{type(self.model.backbone)}}')\n",
            f"{indent}# print(f'DEBUG: Attributes: {{dir(self.model.backbone)}}')\n",
            f"{indent}if hasattr(self.model.backbone, 'visual'):\n",
            f"{indent}    self.image_encoder = self.model.backbone.visual\n",
            f"{indent}elif hasattr(self.model.backbone, 'trunk'):\n",
            f"{indent}    print('DEBUG: Found .trunk, using it as image_encoder')\n",
            f"{indent}    self.image_encoder = self.model.backbone.trunk\n",
            f"{indent}else:\n",
            f"{indent}    print('DEBUG: No .visual or .trunk found. Using backbone itself as image_encoder')\n",
            f"{indent}    self.image_encoder = self.model.backbone\n",
            f"{indent}# --- FIX BY GEMINI END ---\n"
        ]
        new_lines.extend(replacement)
    else:
        new_lines.append(line)

if found:
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    print("✅ Successfully patched monoclip.py!")
else:
    print("⚠️ Warning: Could not find the target line to replace.")
    print("Check if the file was already modified.")
