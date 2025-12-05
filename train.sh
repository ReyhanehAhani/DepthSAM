#!/bin/bash
#SBATCH --job-name=SAM3_Train_Adapter
#SBATCH --output=slurm_logs_train/train_%j.out
#SBATCH --error=slurm_logs_train/train_%j.err
#SBATCH --time=03:00:00
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

# 2. لود ماژول‌ها (بهترین حالت برای H100)
echo "==== Purging and Loading Modules ===="
module --force purge
module load StdEnv/2023
module load gcc
module load cuda
module load opencv # برای حل مشکل cv2

# 3. تنظیمات محیطی
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export BLIS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

# 4. فعال‌سازی محیط مجازی
source /home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate

# 5. مسیر کتابخانه‌ها
export PYTHONPATH=/scratch/ram112/python_libs:$PYTHONPATH

# 6. رفتن به پوشه کد
cd /home/ram112/projects/def-jieliang/ram112/All_DEPTHCLIP/DepthCLIP/DepthCLIP_code

# 7. اجرای آموزش با مسیر SCRATCH
echo "==== Starting Training ===="
python3 train.py \
  --batch_size 8 \
  --epochs 50 \
  --lr 1e-3 \
  --data_path "/scratch/ram112/NYU_dataset" \
  --trainfile_nyu "./datasets/my_test_list.txt" \
  --save_path "/scratch/ram112/checkpoints_trained" 

echo "==== Job finished at: $(date) ===="