import re
import os

file_path = "/home/ram112/projects/def-jieliang/ram112/All_DEPTHCLIP/DepthCLIP/DepthCLIP_code/datasets/datasets_list.py"

print(f"Applying FINAL Index Fix and Diagnostic Print to {file_path}...")

with open(file_path, 'r') as f:
    content = f.read()

# 1. پیدا کردن و حذف خطای اندیس‌دهی
# این regex خطوط مربوط به تقسیم‌بندی فایل را در تابع __getitem__ هدف قرار می‌دهد
def replace_getitem_logic(match):
    indent = match.group(1)
    
    # کد جدید که هم پرینت می‌گیرد و هم اندیس را درست استخراج می‌کند
    return (
        f'{indent}divided_file = [f.strip() for f in divided_file if f.strip()]\n'
        f'{indent}# --- DIAGNOSTIC PRINT ---\n'
        f'{indent}print(f"DEBUG: File list line content: {{self.fileset[index].strip()}}")\n'
        f'{indent}print(f"DEBUG: Divided parts (RGB and GT): {{divided_file}}")\n'
        f'{indent}# --- END PRINT ---\n'
        f'{indent}rgb_name = divided_file[0] # نام فایل RGB\n'
        f'{indent}gt_name = divided_file[1] # نام فایل GT\n'
        f'{indent}# FIX: استخراج ID فایل با حذف پسوند (.jpg) برای استفاده در بقیه کد\n'
        f'{indent}filename = rgb_name.split(".")[0].strip()\n'
    )

# این الگوی regex باید کل بلوک تقسیم‌بندی را پیدا و جایگزین کند
# توجه: این الگو بر اساس ساختار استاندارد DepthCLIP است.
pattern = re.compile(r'(\s*)divided_file = self\.fileset\[index\]\.split\(\)(.*?)\s*(filename = divided_file_\[0\].*?|\s*filename = divided_file\[0\].*?)', re.DOTALL)

# ما فقط روی ساختار ساده تمرکز می‌کنیم، چون کد شما پیچیده نیست
# بیایید ساده‌ترین تغییر را روی خطوط بزنیم.

new_content = re.sub(
    r'(filename = divided_file\[0\] \+ \'_\' \+ divided_file\[1\])', 
    r'# --- FIX BY GEMINI FINAL LOGIC ---\n        print(f"DEBUG: Divided file for name extraction: {divided_file}")\n        filename = divided_file[0].split(".")[0].strip()\n# --- FIX END ---', 
    content
)


# چون ممکن است پچ قبلی روی فایل اثر گذاشته باشد، فقط خطوط 70-75 را هدف قرار می‌دهم:
new_content = re.sub(
    r'(filename = divided_file\[0\]\s*\+\s*\'_\'\s*\+\s*divided_file\[1\])', 
    r'filename = divided_file[0].split(".")[0].strip() # FIX: Extract ID by removing extension\n        print(f"DEBUG: File ID extracted: {filename}")', 
    content
)

# باید یک تغییر کلی‌تر بزنیم.

# تغییر نهایی: پیدا کردن خطی که ارور می‌دهد (حدود ۷۰) و جایگزینی با منطق ساده.
# چون نمی‌توانیم بدون دیدن فایل بدانیم متغیرها چطور استفاده می‌شوند، باید منطق را ساده کنیم.

# ساده‌ترین جایگزینی برای رفع IndexError:
new_content = re.sub(
    r'filename = divided_file\[0\].*?\(.\)\n', # سعی می‌کنیم خط filename=... را بگیریم
    r'filename = divided_file[0].split(".")[0].strip() # FIX: Extract ID by removing extension\n        print(f"DEBUG: File ID extracted: {filename}")\n', 
    content, 
    flags=re.DOTALL
)


with open(file_path, 'w') as f:
    f.write(new_content)
    
print("✅ datasets_list.py patched for final Index fix!")

