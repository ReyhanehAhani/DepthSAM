#!/bin/bash
#SBATCH --job-name=SAM3_Final_Run
#SBATCH --output=slurm_logs_train/train_CNN_%j.out
#SBATCH --error=slurm_logs_train/train_CNN_%j.err
#SBATCH --time=05:00:00
#SBATCH --account=def-jieliang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

# 1. ساخت پوشه لاگ
mkdir -p slurm_logs_train

echo "==== Job started at: $(date) ===="
echo "Node: $(hostname)"

# 2. لود ماژول‌ها (بهترین ترکیب برای Compute Canada)
echo "==== Purging and Loading Modules ===="
module --force purge
module load StdEnv/2023
module load gcc
module load cuda
module load opencv 

# 3. تنظیمات محیطی حیاتی (برای جلوگیری از کرش CPU)
# نکته: وقتی DataLoader پایتون کار می‌کند، اینها باید 1 باشند
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# نکته حیاتی: این خط را کامنت کردیم تا SLURM خودش GPU را مدیریت کند
# export CUDA_VISIBLE_DEVICES=0 

# 4. فعال‌سازی محیط مجازی
source /home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate

# 5. مسیر کتابخانه‌ها
export PYTHONPATH=/scratch/ram112/python_libs:$PYTHONPATH

# 6. رفتن به پوشه کد
cd /home/ram112/projects/def-jieliang/ram112/All_DEPTHCLIP/DepthCLIP/DepthCLIP_code

# 7. اجرای آموزش
echo "==== Starting Training ===="

# Batch Size روی 4 تنظیم شده تا مموری H100 پر نشود
python3 train.py \
  --batch_size 4 \
  --epochs 50 \
  --lr 1e-4 \
  --data_path "/scratch/ram112/NYU_dataset" \
  --trainfile_nyu "./datasets/my_test_list.txt" \
  --save_path "/scratch/ram112/checkpoints_trained" 

echo "==== Job finished at: $(date) ===="