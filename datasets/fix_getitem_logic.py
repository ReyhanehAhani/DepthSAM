import re

path = "datasets_list.py"
print("Applying final fix by replacing the __getitem__ logic...")

with open(path, 'r') as f:
    content = f.read()

# این الگو کل تابع __getitem__ را هدف قرار می‌دهد
# و منطق استخراج نام فایل را در آن تغییر می‌دهد.
# (من این را برای کار کردن با ساختار کد شما تطبیق دادم)
new_content = re.sub(
    r'(def __getitem__\(self, index\):.*?)(rgb = Image\.open\(rgb_file\))',
    r'''\1
        # --- FIX BY GEMINI FINAL LOGIC (IndexError) ---
        divided_file = [f.strip() for f in self.fileset[index].split() if f.strip()]
        
        # NOTE: ما مطمئن می‌شویم که دو فایل (RGB و GT) در لیست وجود دارند
        if len(divided_file) < 2:
            print(f"ERROR: List line is corrupted: {self.fileset[index]}")
            # در محیط کلاستر، اجازه می‌دهیم کرش کند تا خطا مشخص شود
            raise IndexError("List item incomplete (Expected RGB and GT filenames)")
        
        rgb_name = divided_file[0]
        gt_name = divided_file[1]
        
        # استخراج ID فایل با حذف پسوند (.jpg) برای استفاده در بقیه کد
        filename = rgb_name.split(".")[0].strip()
        
        rgb_file = os.path.join(self.data_path, rgb_name)
        gt_file = os.path.join(self.data_path, gt_name)
        # --- FIX END ---
        
        rgb = Image.open(rgb_file)''',
    content, 
    flags=re.DOTALL
)

# چون regex بالا پیچیده است، اگر شکست خورد، فقط خط filename را جایگزین می‌کنیم:
if not re.search(r'# --- FIX BY GEMINI FINAL LOGIC ---', new_content):
    print("WARNING: Complex regex failed. Trying simpler replacement...")
    new_content = re.sub(
        r'(filename = divided_file\[0\] \+ \'_\' \+ divided_file\[1\])', 
        r'filename = divided_file[0].split(".")[0].strip()', 
        content
    )
    
with open(path, 'w') as f:
    f.write(new_content)
    
print("✅ datasets_list.py patched for final Index fix!")

