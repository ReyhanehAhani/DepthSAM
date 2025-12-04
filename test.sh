#!/bin/bash
#SBATCH --job-name=SAM3_Inference
#SBATCH --output=slurm_logs_infer/sam3_final_%j.out
#SBATCH --error=slurm_logs_infer/sam3_final_%j.err
#SBATCH --time=01:00:00
#SBATCH --account=def-jieliang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

# 1. ساخت پوشه لاگ
mkdir -p slurm_logs_infer

echo "==== Job started at: $(date) ===="
echo "Node: $(hostname)"

# 2. FIX 1: پاک‌سازی و لود ماژول‌های ضروری
echo "==== Purging and Loading Modules ===="
module --force purge
module load StdEnv/2023
module load gcc cuda 
module load opencv

# 3. FIX 2: کنترل Threading برای جلوگیری از BLIS Abort
# این خطوط تضمین می‌کنند که BLAS/NumPy/PyTorch فقط از 12 هسته‌ی درخواستی استفاده کنند.
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export BLIS_NUM_THREADS=12
export CUDA_VISIBLE_DEVICES=0 # اجبار به دیدن GPU 0

# 4. فعال‌سازی محیط مجازی
echo "==== Activating Virtual Environment ===="
source /home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate

# 5. تنظیم مسیر کتابخانه Triton/Decord/pycocotools
echo "==== Setting PYTHONPATH ===="
export PYTHONPATH=/scratch/ram112/python_libs:$PYTHONPATH

# 6. رفتن به پوشه کد
cd /home/ram112/projects/def-jieliang/ram112/All_DEPTHCLIP/DepthCLIP/DepthCLIP_code

# 7. اجرای تست
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