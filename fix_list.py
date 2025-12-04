import os

# مسیر لیست قدیمی (که پوشه‌بندی دارد)
old_list_path = "/home/ram112/projects/def-jieliang/ram112/All_DEPTHCLIP/DepthCLIP/DepthCLIP_code/datasets/nyudepthv2_test_files_with_gt_dense.txt"

# مسیر لیست جدید (که می‌خواهیم بسازیم)
new_list_path = "/home/ram112/projects/def-jieliang/ram112/All_DEPTHCLIP/DepthCLIP/DepthCLIP_code/datasets/my_test_list.txt"

# مسیر واقعی عکس‌های ما در Scratch
data_dir = "/scratch/ram112/NYU_dataset"

print("🔄 Converting dataset list to match flat directory structure...")

new_lines = []
missing_count = 0
found_count = 0

try:
    with open(old_list_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2: continue

        # اسم فایل‌ها را از مسیر طولانی بیرون می‌کشیم
        # مثلا: 'bathroom/rgb_00045.jpg' -> 'rgb_00045.jpg'
        img_name = os.path.basename(parts[0])
        depth_name = os.path.basename(parts[1])

        # چک می‌کنیم آیا این فایل واقعا در اسکرچ هست؟
        if os.path.exists(os.path.join(data_dir, img_name)):
            # اگر بود، خط جدید را می‌نویسیم: 'rgb_00045.jpg sync_depth_00045.png'
            new_lines.append(f"{img_name} {depth_name}\n")
            found_count += 1
        else:
            # اگر نبود، یعنی ایندکس‌ها نمی‌خوانند
            missing_count += 1
            if missing_count < 5:
                print(f"⚠️ Missing file: {img_name}")

    # ذخیره لیست جدید
    with open(new_list_path, 'w') as f:
        f.writelines(new_lines)

    print("-" * 30)
    print(f"✅ Created new list: {new_list_path}")
    print(f"📊 Found files: {found_count}")
    print(f"📉 Missing files: {missing_count}")
    
    if found_count > 0:
        print("🚀 Ready to run test.sh with the new list!")

except FileNotFoundError:
    print("❌ Error: Original list file not found.")

