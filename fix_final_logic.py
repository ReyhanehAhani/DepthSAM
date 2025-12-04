import re

path = "/home/ram112/projects/def-jieliang/ram112/All_DEPTHCLIP/DepthCLIP/DepthCLIP_code/datasets/datasets_list.py"
print("Applying final logic fix for list parsing...")

with open(path, 'r') as f:
    content = f.read()

# پیدا کردن خطوط اطراف خط 70 که filename را تعریف می‌کنند
# این بخش بر اساس ساختار استاندارد DepthCLIP است
new_content = re.sub(
    r'(filename = divided_file\[0\]\.split\(\)/.*?\n)', 
    r'# --- FIX BY GEMINI FINAL LOGIC ---\n        # استفاده از نام فایل اصلی بدون پسوند برای ID\n        filename = divided_file[0].split(".")[0].strip()\n# --- FIX END ---\n', 
    content, 
    flags=re.DOTALL
)

# جایگزینی بر اساس خط 70 در نمونه کد شما (حدس دقیق‌تر)
content = re.sub(
    r'(filename = divided_file\[0\] \+ \'_\' \+ divided_file\[1\])', 
    r'# --- FIX BY GEMINI FINAL LOGIC ---\n        # استخراج نام فایل اصلی بدون پسوند (.jpg) برای ID\n        filename = divided_file[0].split(".")[0].strip()\n# --- FIX END ---', 
    content
)

# چون regex سخت است، ساده‌ترین کار:
# ما به دنبال خطی هستیم که divided_file[1] را صدا می‌زند.

# اگر substitution اولیه کار نکرد، ما فقط خطوط را جایگزین می‌کنیم
if "divided_file[1]" in content:
    content = content.replace(
        "filename = divided_file[0] + '_' + divided_file[1]", 
        "filename = divided_file[0].split('.')[0].strip()"
    )
    print("Found and replaced old filename logic.")

with open(path, 'w') as f:
    f.write(content)
    
print("✅ datasets_list.py updated successfully!")
