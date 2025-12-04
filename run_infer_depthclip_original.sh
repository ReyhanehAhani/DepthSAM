#!/bin/bash
#SBATCH --job-name=infer_depthclip_orig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=def-jieliang
#SBATCH --output=06-October/infer_depthclip_orig_%j.out
#SBATCH --error=06-October/infer_depthclip_orig_%j.err

# اطمینان از ساخت پوشه لاگ
mkdir -p logs_finals/06-October

echo "==== Loading OpenCV module ===="
module load gcc opencv

echo "==== Activating virtual environment ===="
source /home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate

echo "--- Python Debug Info ---"
which python
python -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
echo "--------------------------"

cd /home/ram112/projects/def-jieliang/ram112/DepthCLIP/DepthCLIP_code

echo "==== Running DepthCLIP original inference ===="
python inference_depthclip_original.py \
  --data_root /project/6006955/ram112/nyu_data \
  --num_samples 200 \
  --out_dir ./viz_outputs_depthclip_original \
  --cmap turbo

echo "==== Job finished at: $(date) ===="
