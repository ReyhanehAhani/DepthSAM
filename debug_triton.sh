#!/bin/bash
#SBATCH --job-name=Debug_Triton
#SBATCH --output=debug_triton_%j.out
#SBATCH --error=debug_triton_%j.err
#SBATCH --time=00:15:00
#SBATCH --account=def-jieliang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

echo "==== Job started at: $(date) ===="

# لود کردن ماژول‌ها
module load gcc opencv

# فعال‌سازی محیط
source /home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate

# اضافه کردن مسیر کتابخانه‌های اسکرچ (حیاتی!)
export PYTHONPATH=/scratch/ram112/python_libs:$PYTHONPATH

echo "==== Testing Triton Import ===="
# اجرای تست ایمپورت با چاپ جزئیات بیشتر
python -c "
import sys
import os

print(f'PYTHONPATH: {os.environ.get(\"PYTHONPATH\", \"Not Set\")}')

try:
    import triton
    print(f'✅ Triton imported from: {triton.__file__}')
    
    import triton.runtime
    print('✅ Triton Runtime imported successfully')
    
except ImportError as e:
    print(f'❌ Import Error: {e}')
    print('Search path was:', sys.path)
except Exception as e:
    print(f'❌ Other Error: {e}')
"

echo "==== Job finished at: $(date) ===="