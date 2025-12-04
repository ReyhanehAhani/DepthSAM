#!/bin/bash
#SBATCH --job-name=SAM3_Inference
#SBATCH --output=slurm_logs_infer/sam3_final_%j.out
#SBATCH --error=slurm_logs_infer/sam3_final_%j.err
#SBATCH --time=01:00:00
#SBATCH --account=def-jieliang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

# 1. ساخت پوشه لاگ
mkdir -p slurm_logs_infer

echo "==== Job started at: $(date) ===="
echo "Node: $(hostname)"

# 2. لود ماژول‌ها
echo "==== Purging and Loading Modules ===="
module --force purge
# استفاده از StdEnv جدیدتر برای سازگاری بهتر با H100 و Python 3.12
module load StdEnv/2023
module load gcc
module load cuda
module load opencv # <--- FIX CRITICAL: بازگرداندن این خط برای رفع مشکل Dummy Package

# 3. تنظیم Threading (برای جلوگیری از کرش BLIS/Numpy)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export BLIS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

# 4. فعال‌سازی محیط مجازی
echo "==== Activating Virtual Environment ===="
source /home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate

# 5. چک کردن وضعیت OpenCV (دیباگ)
echo "==== Checking OpenCV Import ===="
# این خط چک می‌کند که آیا cv2 قبل از اجرای اصلی کار می‌کند یا خیر
python3 -c "import cv2; print(f'OpenCV Version: {cv2.__version__}')" || echo "WARNING: OpenCV import failed in check, but proceeding..."

# 6. تنظیم مسیر کتابخانه‌ها
echo "==== Setting PYTHONPATH ===="
export PYTHONPATH=/scratch/ram112/python_libs:$PYTHONPATH

# 7. رفتن به پوشه کد
cd /home/ram112/projects/def-jieliang/ram112/All_DEPTHCLIP/DepthCLIP/DepthCLIP_code

# 8. اجرای تست
echo "==== Running Inference with SAM 3 ===="
python3 eval.py \
  --evaluate \
  --batch_size 1 \
  --dataset NYU \
  --gpu_num 0 \
  --other_method 'MonoCLIP' \
  --testfile_nyu "/home/ram112/projects/def-jieliang/ram112/All_DEPTHCLIP/DepthCLIP/DepthCLIP_code/datasets/my_test_list.txt" \
  --data_path "/scratch/ram112/NYU_dataset"

echo "==== Job finished at: $(date) ===="